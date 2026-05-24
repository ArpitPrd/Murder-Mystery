#!/usr/bin/env bash
# setup.sh — One-shot environment setup for Project OMNISCIENT.
#
# What this script does:
#   1. Installs Ollama (Linux/macOS).
#   2. Starts the Ollama service (if not already running).
#   3. Pulls the detective LLM: qwen2.5:7b (~4.7 GB download).
#   4. Installs Python runtime dependencies.
#   5. Creates a .env file from .env.example if one does not exist.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# After running this script:
#   1. Edit .env and set GEMINI_API_KEY=<your key>.
#   2. Run the system: python main.py 0

set -e

DETECTIVE_MODEL="${DETECTIVE_MODEL:-qwen2.5:7b}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

echo "=== Project OMNISCIENT — Environment Setup ==="
echo ""

# ----------------------------------------------------------------
# Step 1: Install Ollama
# ----------------------------------------------------------------
if command -v ollama &>/dev/null; then
    echo "[OK] Ollama is already installed: $(ollama --version)"
else
    echo "[*] Installing Ollama..."
    if [[ "$(uname)" == "Linux" ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    elif [[ "$(uname)" == "Darwin" ]]; then
        if command -v brew &>/dev/null; then
            brew install ollama
        else
            echo "[!] Homebrew not found. Download Ollama from https://ollama.com/download/mac"
            exit 1
        fi
    else
        echo "[!] Unsupported OS. Download Ollama manually from https://ollama.com/download"
        exit 1
    fi
    echo "[OK] Ollama installed."
fi

# ----------------------------------------------------------------
# Step 2: Start Ollama service (runs in background if not active)
# ----------------------------------------------------------------
if curl -s "${OLLAMA_HOST}/api/tags" &>/dev/null; then
    echo "[OK] Ollama service is already running at ${OLLAMA_HOST}"
else
    echo "[*] Starting Ollama service in background..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
    if curl -s "${OLLAMA_HOST}/api/tags" &>/dev/null; then
        echo "[OK] Ollama service started."
    else
        echo "[!] Could not start Ollama. Check /tmp/ollama.log for details."
        exit 1
    fi
fi

# ----------------------------------------------------------------
# Step 3: Pull the detective model
# ----------------------------------------------------------------
PULLED_MODELS=$(ollama list 2>/dev/null | awk '{print $1}')
MODEL_TAG="${DETECTIVE_MODEL%%:*}:${DETECTIVE_MODEL##*:}"

if echo "$PULLED_MODELS" | grep -q "^${MODEL_TAG}"; then
    echo "[OK] Model '${DETECTIVE_MODEL}' is already pulled."
else
    echo "[*] Pulling detective model: ${DETECTIVE_MODEL} (~4.7 GB)..."
    echo "    This may take several minutes depending on your connection."
    ollama pull "${DETECTIVE_MODEL}"
    echo "[OK] Model '${DETECTIVE_MODEL}' pulled."
fi

# ----------------------------------------------------------------
# Step 4: Install Python dependencies
# ----------------------------------------------------------------
echo "[*] Installing Python runtime dependencies..."
pip install -r requirements.txt --quiet
echo "[OK] Python dependencies installed."

# ----------------------------------------------------------------
# Step 5: Create .env from template if needed
# ----------------------------------------------------------------
if [ -f ".env" ]; then
    echo "[OK] .env file already exists. Skipping."
else
    cp .env.example .env
    echo "[OK] Created .env from .env.example."
    echo ""
    echo "ACTION REQUIRED: Open .env and set your GEMINI_API_KEY before running."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your GEMINI_API_KEY"
echo "  2. Run a case:    python main.py 0"
echo "  3. Run case 42:   python main.py 42"
echo ""
echo "Optional: fine-tune the local parser model on the dataset:"
echo "  pip install -r requirements-finetune.txt"
echo "  python generate_training_data.py"
echo "  python generate_training_data.py --split val"
echo "  python train_parser.py"
