import argparse
import json
import logging
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
 
# LoRA rank. Higher = more capacity but more VRAM and slower training.
# 16 is a safe default for a task this focused.
LORA_RANK = 16
 
# Maximum sequence length. Parser inputs are short; 1024 is sufficient.
MAX_SEQ_LENGTH = 1024
 
# Training data paths
TRAIN_PATH = "training_data/train.jsonl"
VAL_PATH   = "training_data/val.jsonl"
 
# Output directory for the fine-tuned model
DEFAULT_OUTPUT = "parser_model_finetuned"


def load_jsonl(path: str) -> list:
    """Load a JSONL file and return a list of dicts."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples

def format_example(example: dict, tokenizer) -> str:
    """
    Format one training example as a single string using the tokenizer's
    chat template. Most instruction-tuned models have a built-in template
    (e.g. <|im_start|>system ... <|im_end|>) that should be used.
 
    Args:
        example   : Dict with 'messages' key in chat format.
        tokenizer : HuggingFace tokenizer with apply_chat_template.
 
    Returns:
        Formatted string ready for tokenisation.
    """
 
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def train(base_model: str, output_dir: str, epochs: int, batch_size: int, learning_rate: float, ) -> None:
    # Step 1 — load base model with 4-bit quantisation
    logger.info("Loading base model: %s", base_model)
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error(
            "unsloth is not installed. Run: pip install unsloth"
        )
        raise
 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,       # 4-bit quantisation for VRAM efficiency
        dtype=None,              # auto-detect: float16 on Ampere, bfloat16 on newer
    )
    logger.info("Base model loaded.")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,    # alpha = rank is a safe default
        lora_dropout=0.0,        # 0.0 is optimal for most LoRA fine-tunes
        bias="none",
        use_gradient_checkpointing="unsloth",  # reduces VRAM use during backprop
        random_state=42,
    )
    logger.info("LoRA adapters attached | rank=%d", LORA_RANK)

    train_raw = load_jsonl(TRAIN_PATH)
    val_raw   = load_jsonl(VAL_PATH)
 
    # Format using the model's chat template
    train_texts = [format_example(ex, tokenizer) for ex in train_raw]
    val_texts   = [format_example(ex, tokenizer) for ex in val_raw]
 
    from datasets import Dataset
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset   = Dataset.from_dict({"text": val_texts})
 
    logger.info(
        "Datasets ready | train=%d | val=%d", len(train_dataset), len(val_dataset)
    )

    try:
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        logger.error("trl is not installed. Run: pip install trl")
        raise

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,   # effective batch = batch_size * 4
        warmup_steps=10,
        learning_rate=learning_rate,
        fp16=True,                        # mixed precision training
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",                 # disable wandb/tensorboard unless configured
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )
 
    logger.info("Starting training | epochs=%d | batch_size=%d | lr=%s",
                epochs, batch_size, learning_rate)
    trainer.train()
    logger.info("Training complete.")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to %s", output_dir)

    try:
        model.save_pretrained_gguf(
            output_dir + "_gguf",
            tokenizer,
            quantization_method="q4_k_m",  # good balance of size and quality
        )
        logger.info("GGUF model saved to %s_gguf", output_dir)
        logger.info(
            "To serve via ollama: ollama create parser_model -f %s_gguf/Modelfile",
            output_dir,
        )
    except Exception as exc:
        logger.warning("GGUF export failed (non-critical): %s", exc)

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a local LLM as the OMNISCIENT parser."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Base model to fine-tune (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output directory for the fine-tuned model (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Per-device training batch size (default: 2).",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate (default: 2e-4).",
    )
    args = parser.parse_args()
 
    # Validate training data exists
    import os
    if not os.path.exists(TRAIN_PATH):
        logger.error(
            "Training data not found at %s. Run generate_training_data.py first.",
            TRAIN_PATH,
        )
        raise FileNotFoundError(TRAIN_PATH)
    if not os.path.exists(VAL_PATH):
        logger.error(
            "Validation data not found at %s. Run generate_training_data.py --split val first.",
            VAL_PATH,
        )
        raise FileNotFoundError(VAL_PATH)
 
    train(
        base_model=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
 
 
if __name__ == "__main__":
    main()
 












