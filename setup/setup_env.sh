#!/usr/bin/env bash
# ======================================================================
# setup/setup_env.sh
#
# One-shot environment setup for the course
#   "From Single Agents to Multi-Agent Systems"
#
# What it does (idempotent — safe to re-run):
#   1. Creates a Python 3.11+ virtual environment in ./.venv (if missing)
#   2. Upgrades pip inside that venv
#   3. Installs everything in setup/requirements.txt
#   4. Registers the venv as a Jupyter kernel ("ising-agents")
#   5. Verifies Ollama is installed, pulls qwen3.5:4b if not present
#   6. Downloads the arXiv paper corpus (data/papers/) if empty
#
# Usage:
#   ./setup/setup_env.sh                  # default model: qwen3.5:4b
#   ISING_OLLAMA_MODEL=qwen3.5:9b ./setup/setup_env.sh
#   SKIP_PAPERS=1 ./setup/setup_env.sh    # don't hit arXiv
#
# After running, activate the venv with:  source .venv/bin/activate
# ======================================================================

set -euo pipefail

# Always run from the repo root (one level above this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
OLLAMA_MODEL="${ISING_OLLAMA_MODEL:-qwen3.5:4b}"
SKIP_PAPERS="${SKIP_PAPERS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- pretty logging helpers -------------------------------------------
log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  !\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m  ✗\033[0m %s\n' "$*" >&2; exit 1; }

# ======================================================================
# 1. Python venv
# ======================================================================
log "Checking Python interpreter..."
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    fail "'$PYTHON_BIN' not found on PATH. Install Python 3.11+ first."
fi
PY_VERSION=$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    warn "Python $PY_VERSION detected. The course targets 3.11+. Proceeding anyway."
else
    ok "Python $PY_VERSION"
fi

log "Virtual environment at $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    ok "Created $VENV_DIR"
else
    ok "Already exists"
fi

# From here on, always use the venv's python/pip explicitly.
VENV_PY="$VENV_DIR/bin/python"
if [ ! -x "$VENV_PY" ]; then
    fail "Expected $VENV_PY to exist after venv creation."
fi

# ======================================================================
# 2. Dependencies
# ======================================================================
log "Upgrading pip inside the venv..."
"$VENV_PY" -m pip install --upgrade pip --quiet
ok "pip up to date"

REQ_FILE="setup/requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
    fail "Missing $REQ_FILE"
fi
log "Installing $REQ_FILE (this can take several minutes on first run)..."
# --upgrade ensures rerunning picks up any pinned-version changes.
"$VENV_PY" -m pip install --upgrade -r "$REQ_FILE"
ok "Dependencies installed"

# ======================================================================
# 3. Jupyter kernel
# ======================================================================
log "Registering Jupyter kernel 'ising-agents'..."
"$VENV_PY" -m ipykernel install --user --name ising-agents \
    --display-name "Python (ising-agents)" >/dev/null
ok "Kernel 'ising-agents' registered"

# ======================================================================
# 4. Ollama
# ======================================================================
log "Checking Ollama..."
if ! command -v ollama >/dev/null 2>&1; then
    warn "Ollama is not installed or not on PATH."
    warn "Install from https://ollama.com/download and re-run this script."
    warn "(Skipping the model pull for now.)"
else
    ok "ollama binary: $(command -v ollama)"

    # Make sure the daemon is actually reachable. Don't try to start it
    # automatically — on macOS the GUI launcher is the right path.
    if ollama list >/dev/null 2>&1; then
        ok "Ollama daemon reachable"
        if ollama list | awk 'NR>1 {print $1}' | grep -qx "$OLLAMA_MODEL"; then
            ok "Model $OLLAMA_MODEL already present"
        else
            log "Pulling model $OLLAMA_MODEL (this is a multi-GB download)..."
            ollama pull "$OLLAMA_MODEL"
            ok "Pulled $OLLAMA_MODEL"
        fi
    else
        warn "Ollama daemon not reachable. Start it with 'ollama serve' or"
        warn "by launching the Ollama app, then re-run this script."
    fi
fi

# ======================================================================
# 5. Paper corpus
# ======================================================================
if [ "$SKIP_PAPERS" = "1" ]; then
    log "SKIP_PAPERS=1 — not downloading the arXiv corpus."
else
    log "Downloading arXiv paper corpus..."
    # Count existing PDFs (glob may not match, so use find).
    NUM_PDFS=$(find data/papers -maxdepth 1 -name '*.pdf' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$NUM_PDFS" -ge 5 ]; then
        ok "data/papers already has $NUM_PDFS PDFs — skipping"
    else
        "$VENV_PY" setup/download_papers.py
    fi
fi

# ======================================================================
# Done
# ======================================================================
echo
log "Setup complete."
log "Next steps:"
log "  1. source $VENV_DIR/bin/activate"
log "  2. python setup/verify_setup.py"
log "  3. jupyter lab       # and select the 'Python (ising-agents)' kernel"
