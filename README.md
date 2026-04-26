# From Single AI Agent to Multi-Agent Systems for Science

Hands-on course (3 days, MHPC 2026): building single-agent and multi-agent LLM systems with the 2D Ising model as the running physics use case.

## Course Abstract

LLMs are rapidly evolving from passive text generators into autonomous agents capable of reasoning, using tools, and collaborating with each other. This hands-on course bridges the gap between understanding LLMs and building functional AI agent systems, using a single physics use case (the phase transition of the 2D Ising model) as a concrete use case. In the first part, the focus is on a single agent. We will incrementally construct a Physics Research Assistant from scratch. Starting from raw API calls to a local LLM, each module adds exactly one capability: step-by-step reasoning, grounding in scientific literature through RAG, execution of Monte Carlo simulations via standardized tool protocols, and self-correction through reflection mechanisms. Every component is verified against the model's exact analytical solutions. In the second part, the focus will be on multi-agent systems. The single agent is decomposed into a team of specialized agents (a Theorist, an Experimentalist, and a Scholar) each with distinct roles and tools. We will explore three increasingly rigorous coordination architectures on the same benchmark task: a sequential pipeline, an adversarial debate, and a typed message bus with machine-verifiable isolation guarantees. Each architecture is compared on identical inputs, making trade-offs between role clarity, error detection, and reliability directly observable. The course surfaces real failure modes of small local models and teaches practical engineering techniques to mitigate them.

---

## Prerequisites

- **Python 3.11+**
- **Ollama** — install from [ollama.com/download](https://ollama.com/download)
- **Git**
- A laptop with **≥16 GB RAM** is strongly recommended (the course is built against `qwen2.5:7b` which needs ~5 GB of VRAM/RAM to run comfortably).

---

## Installation

### 1. Clone the repo

```bash
git clone <course-repo-url>
cd ising-agents-course
```

### 2. Start Ollama and pull the course model

```bash
# On a separate terminal (Ollama must keep running during the lectures):
ollama serve

# Back in your normal terminal:
ollama pull qwen2.5:7b

# We migh also play with:
ollama pull llama3.1:8b
```

Verify Ollama answers:

```bash
ollama run qwen2.5:7b "Say hello."
```

### 3. Create the Python virtual environment and install dependencies

There are two ways — pick one.

**Option A (recommended): the one-shot setup script**

```bash
./setup/setup_env.sh
```

This creates `.venv/`, installs `setup/requirements.txt`, registers a Jupyter kernel named `ising-agents`, and downloads the paper corpus (~8 arXiv PDFs on the 2D Ising model) into `data/papers/`.

**Option B: manual**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r setup/requirements.txt
python setup/download_papers.py
```

### 4. Verify the setup

```bash
source .venv/bin/activate   # if not already active
python setup/verify_setup.py
```

This checks Ollama is reachable, the model responds, all Python packages import, and the paper corpus is present. Fix any red line before the lectures start.

---

## Running the notebooks

```bash
source .venv/bin/activate
jupyter lab
```

Open `notebooks/` and start from `01_naked_llm.ipynb`. Each notebook is self-contained and numbered in the order we run them during the lectures:

- **SA Part** (single-agent): notebooks `01`–`06`
- **MAS Part** (multi-agent): notebooks `07`–`10`

If Jupyter doesn't pick up the course environment automatically, select the **`ising-agents`** kernel from the kernel menu.

---

## Troubleshooting

- **`ollama: command not found`** — install Ollama from [ollama.com/download](https://ollama.com/download) and make sure `ollama serve` is running.
- **`ConnectionError` when an agent calls the model** — Ollama is not running. Start it with `ollama serve` in a separate terminal.
- **`ModuleNotFoundError`** in a notebook — you're on the wrong kernel. Select `ising-agents` from Jupyter's kernel menu.
- **The model is very slow / laptop overheats** — try a smaller model: `ollama pull qwen2.5:3b` and set `LLM_MODEL = "qwen2.5:3b"` where the notebook configures it. The MAS Part modules (especially 2.1–2.3) will surface more failure modes on smaller models — that is expected and pedagogically useful.
- **Paper corpus missing / empty** — rerun `python setup/download_papers.py`.
