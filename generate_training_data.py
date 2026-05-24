"""
Generates fine-tuning training data for the parser LLM.

Strategy:
    For each case in the training split:
        For each suspect:
            Ask a fixed set of questions via SuspectAgent
            Capture raw answers
            Run the Gemini parser to extract structured JSON claims
            Save (question, answer) -> claims_json as a training example

Output format is JSONL, one example per line, in the chat message format
expected by most fine-tuning frameworks (unsloth, trl, OpenAI fine-tuning).

Each line:
{
    "messages": [
        {"role": "system",    "content": "<parser_system_prompt>"},
        {"role": "user",      "content": "SUSPECT NAME: ...\nQUESTION: ...\nANSWER: ..."},
        {"role": "assistant", "content": "[{...claims json...}]"}
    ]
}

Usage:
    python generate_training_data.py
    python generate_training_data.py --cases 50   # limit to first N cases
    python generate_training_data.py --split val  # generate validation set instead
"""

import argparse
import json
import logging
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed interrogation questions used for every suspect.
# These cover the four main objectives: alibi, motive, relationship, knowledge.
# Using a fixed set ensures consistent coverage across all training examples.
# ---------------------------------------------------------------------------
FIXED_QUESTIONS = [
    "Where were you at the time of the murder and can anyone confirm that?",
    "What was your relationship with the victim like recently?",
    "Did you witness anything unusual that evening?",
    "Do you know of anyone who might have wanted to harm the victim?",
    "How did you feel about the victim's recent decisions or actions?",
]

# ---------------------------------------------------------------------------
# Parser system prompt — must match exactly what is in interrogator.py
# so the fine-tuned model sees the same prompt at inference time.
# ---------------------------------------------------------------------------
PARSER_SYSTEM_PROMPT = (
    "You are a forensic claim extractor. You receive a question and a suspect's "
    "raw answer. Extract every factual claim as a JSON array.\n\n"
    "Each element must have these exact keys:\n"
    "  claim_type : one of alibi | accusation | denial | relationship | knowledge | inconsistency\n"
    "  subject    : who the claim is about (full name)\n"
    "  predicate  : the relationship or action (snake_case verb phrase)\n"
    "  object_    : what the predicate points to\n"
    "  time_ref   : time reference string or null\n"
    "  confidence : stated | implied | uncertain\n"
    "  source_text: the exact fragment from the answer that grounds this claim\n\n"
    "Return ONLY a valid JSON array. No preamble, no explanation, no markdown fences."
)


def get_suspect_answer(
    suspect_data: dict,
    victim_name: str,
    question: str,
    client: OpenAI,
    model: str,
) -> str:
    """
    Get a single answer from a SuspectAgent for one question.

    Builds the same system prompt as SuspectAgent in suspect_agent.py
    to ensure training data matches the actual interrogation context.

    Args:
        suspect_data : Suspect dict from the dataset.
        victim_name  : Name of the victim.
        question     : Question to ask.
        client       : OpenAI-compatible client.
        model        : Model identifier.

    Returns:
        Raw answer string from the suspect LLM.
    """

    system_prompt = (
        f"You are {suspect_data['name']}. Do not break character.\n\n"
        f"### YOUR BACKGROUND ###\n{suspect_data.get('introduction', '')}\n\n"
        f"### YOUR STORY & TIMELINE ###\n{suspect_data.get('story', '')}\n\n"
        f"### CASE CONTEXT ###\n"
        f"The detective is interrogating you regarding the murder of {victim_name}.\n\n"
        f"### YOUR BEHAVIORAL DIRECTIVE ###\n{suspect_data.get('task', '')}\n"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
        temperature=0.6,
    )
    return response.choices[0].message.content.strip()


def parse_claims(
    suspect_name: str,
    question: str,
    answer: str,
    client: OpenAI,
    model: str,
) -> str:
    """
    Run the parser LLM on a question-answer pair and return the raw JSON string.

    Args:
        suspect_name : Name of the suspect (included in parser input).
        question     : Question that was asked.
        answer       : Suspect's raw answer.
        client       : OpenAI-compatible client.
        model        : Model identifier.

    Returns:
        Raw JSON string from the parser, or None if the call failed.
    """

    user_content = (
        f"SUSPECT NAME: {suspect_name}\n"
        f"QUESTION: {question}\n"
        f"ANSWER: {answer}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PARSER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.0,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


def is_valid_claims_json(raw: str) -> bool:
    """
    Validate that a parser output is a non-empty JSON array with required keys.

    Args:
        raw : Raw string output from the parser LLM.

    Returns:
        True if valid, False otherwise.
    """

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return False
        if len(parsed) == 0:
            return False
        required_keys = {"claim_type", "subject", "predicate", "object_", "confidence", "source_text"}
        for item in parsed:
            if not required_keys.issubset(item.keys()):
                return False
        return True
    except json.JSONDecodeError:
        return False


def build_training_example(
    suspect_name: str,
    question: str,
    answer: str,
    claims_json: str,
) -> dict:
    """
    Build one training example in chat message format.

    Args:
        suspect_name : Name of the suspect.
        question     : Question asked.
        answer       : Suspect's raw answer.
        claims_json  : Validated JSON claims string.

    Returns:
        Dict with 'messages' key in chat format.
    """

    user_content = (
        f"SUSPECT NAME: {suspect_name}\n"
        f"QUESTION: {question}\n"
        f"ANSWER: {answer}"
    )

    return {
        "messages": [
            {"role": "system",    "content": PARSER_SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": claims_json},
        ]
    }


def generate_split(
    cases: list,
    client: OpenAI,
    model: str,
    output_path: str,
    rate_limit_delay: float = 1.0,
) -> None:
    """
    Generate fine-tuning examples for a list of cases and write to JSONL.

    For each case, iterates over all 5 suspects and asks each of the
    FIXED_QUESTIONS. Skips examples where the parser output is invalid JSON.

    Args:
        cases            : List of case dicts.
        client           : OpenAI-compatible client (used for both suspect and parser).
        model            : Model identifier.
        output_path      : Path to write the JSONL output file.
        rate_limit_delay : Seconds to wait between API calls to avoid rate limits.
    """

    total_examples = 0
    skipped = 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_file:
        for case_idx, case in enumerate(cases):
            victim_name = case.get("victim", {}).get("name", "Unknown")
            logger.info(
                "Processing case %d/%d | victim=%s",
                case_idx + 1, len(cases), victim_name,
            )

            for suspect in case.get("suspects", []):
                suspect_name = suspect.get("name", "Unknown")

                for question in FIXED_QUESTIONS:
                    try:
                        # Step 1 — get suspect's answer
                        time.sleep(rate_limit_delay)
                        answer = get_suspect_answer(
                            suspect_data=suspect,
                            victim_name=victim_name,
                            question=question,
                            client=client,
                            model=model,
                        )

                        # Step 2 — parse claims from the answer
                        time.sleep(rate_limit_delay)
                        claims_json = parse_claims(
                            suspect_name=suspect_name,
                            question=question,
                            answer=answer,
                            client=client,
                            model=model,
                        )

                        # Step 3 — validate before saving
                        if not is_valid_claims_json(claims_json):
                            logger.warning(
                                "Skipping invalid parser output | suspect=%s | question=%s",
                                suspect_name, question[:50],
                            )
                            skipped += 1
                            continue

                        # Step 4 — write training example
                        example = build_training_example(
                            suspect_name=suspect_name,
                            question=question,
                            answer=answer,
                            claims_json=claims_json,
                        )
                        out_file.write(json.dumps(example) + "\n")
                        total_examples += 1

                    except Exception as exc:
                        logger.error(
                            "Error on case=%d suspect=%s question=%s | %s",
                            case_idx, suspect_name, question[:50], exc,
                        )
                        skipped += 1
                        continue

            logger.info(
                "Case %d done | total_examples_so_far=%d | skipped=%d",
                case_idx + 1, total_examples, skipped,
            )

    logger.info(
        "Generation complete | output=%s | total=%d | skipped=%d",
        output_path, total_examples, skipped,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate fine-tuning data for the OMNISCIENT parser LLM."
    )
    parser.add_argument(
        "--cases", type=int, default=None,
        help="Limit to first N cases (default: all training cases).",
    )
    parser.add_argument(
        "--split", choices=["train", "val"], default="train",
        help="Which split to generate (default: train).",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between API calls (default: 1.0).",
    )
    args = parser.parse_args()

    # Load dataset
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Split 80/20 — no shuffle to keep reproducibility
    split_idx = int(len(dataset) * 0.8)
    train_cases = dataset[:split_idx]   # 320 cases
    val_cases   = dataset[split_idx:]   # 81 cases

    if args.split == "train":
        cases = train_cases
        output_path = "training_data/train.jsonl"
    else:
        cases = val_cases
        output_path = "training_data/val.jsonl"

    # Apply case limit if specified
    if args.cases:
        cases = cases[:args.cases]
        logger.info("Limited to first %d cases.", args.cases)

    logger.info(
        "Generating %s split | cases=%d | questions_per_suspect=%d | suspects_per_case=5",
        args.split, len(cases), len(FIXED_QUESTIONS),
    )
    logger.info(
        "Expected examples (if no skips): %d",
        len(cases) * 5 * len(FIXED_QUESTIONS),
    )

    # Initialise client
    client = OpenAI(
        api_key=os.environ.get("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )
    model = "gemini-2.5-flash"

    generate_split(
        cases=cases,
        client=client,
        model=model,
        output_path=output_path,
        rate_limit_delay=args.delay,
    )


if __name__ == "__main__":
    main()