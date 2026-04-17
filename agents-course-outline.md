# From Single Agents to Multi-Agent Systems

## A Hands-On Course for Physics, Mathematics & Computer Science Researchers

**Duration:** 2 days × 4 hours (8 hours total)
**Audience:** PhD students and Postdocs in physics, mathematics, neuroscience, and computer science. Background in ML/DL assumed.
**Stack:** Python 3.11+, Ollama (local LLMs), smolagents (HuggingFace), CrewAI, MCP protocol
**Physics scenario:** Studying the phase transition of the 2D Ising model — from literature review to simulation to analysis.

**Philosophy:**
1. Every concept is introduced incrementally — each module adds one capability on top of the previous one.
2. Students always see the code, never just the abstraction.
3. A single coherent physics use case threads through both days: by the end, students have built a multi-agent physics research team that derives predictions, runs simulations, and analyzes results.

---

## Pre-Course Setup

Students must have installed before Day 1:

- **Python 3.11+** with a virtual environment
- **Ollama** with a model pulled: `qwen3.5:4b` (minimum) or `qwen3.5:9b` / `llama3.1:8b` (recommended if ≥16 GB RAM)
- **Python packages:** see `setup/requirements.txt`
- **Course GitHub repo** cloned (contains starter notebooks, MCP server code, paper corpus)

> **Deliverables to build:**
> - `setup/requirements.txt` — all Python dependencies
> - `setup/setup_env.sh` — creates venv, installs packages, pulls Ollama model
> - `setup/download_papers.py` — fetches 5–10 arXiv papers on the 2D Ising model (stores PDFs + metadata in `data/papers/`)
> - `setup/verify_setup.py` — checks Ollama is running, model responds, packages are importable

---

## The Physics Scenario

The **2D Ising model** is the running example for the entire course. It was chosen because:

1. **It's familiar** to physics/math students (most have seen it in statistical mechanics)
2. **It has exact solutions** (Onsager's T_c = 2J / ln(1+√2) ≈ 2.269 J/k_B, exact critical exponents β=1/8, γ=7/4, ν=1) so agents' outputs can be verified quantitatively
3. **It connects to the instructor's research** on Ising LLMs and Naming Game LLMs (Day 2 finale)
4. **The research workflow is complete:** literature → theory → simulation → data analysis → report

---

### DAY 1 — The Single Agent: "Physics Research Assistant"

The Day 1 agent is a **personal research assistant for a PhD student starting a project on the Ising model**. It grows progressively more capable through each module. Here is exactly what it can do after each step, with concrete example interactions:

#### After Module 1.0 (Naked LLM) — The agent is just a chat function

The student can ask physics questions, but the LLM is stateless and has no tools. It can only answer from its training data, may hallucinate, and can't verify anything.

> **Example interaction:**
> - Student: *"What is the critical temperature of the 2D Ising model?"*
> - Agent: *"The critical temperature is T_c = 2J / ln(1+√2) ≈ 2.269 J/k_B, as derived by Onsager in 1944."*
> - This is correct (the LLM memorized it), but the student can't trust it — there's no way to verify, no calculation, no source.

#### After Module 1.1 (ReAct) — The agent can reason step-by-step and use a calculator

The agent now follows the Thought → Action → Observation loop. It has two tools: a `calculator` and a `unit_converter`. It can break down problems, compute intermediate results, and chain reasoning steps.

> **Example interaction:**
> - Student: *"The critical temperature of the 2D Ising model is T_c = 2J/ln(1+√2). If the coupling constant J = 0.5 eV, what is T_c in Kelvin?"*
> - Agent:
>   - Thought: *"I need to compute T_c = 2 × 0.5 / ln(1+√2). First compute ln(1+√2)."*
>   - Action: `calculator("math.log(1 + math.sqrt(2))")` → 0.8814
>   - Thought: *"Now T_c = 1.0 / 0.8814 = 1.1346 eV. I need to convert eV to Kelvin."*
>   - Action: `unit_converter(1.1346, "eV", "K")` → 13,168 K
>   - Final Answer: *"T_c ≈ 13,168 K (about 13,000 K), which is extremely hot — this coupling strength would correspond to a very high-temperature phase transition."*
> - The agent can now *compute*, not just recite.

#### After Module 1.2 (RAG) — The agent can search and cite a corpus of physics papers

The agent now has access to 5–10 Ising model papers stored in a local vector database (ChromaDB). It can retrieve relevant passages and ground its answers in real literature.

> **Example interaction:**
> - Student: *"What is the Wolff algorithm and how does it compare to the Metropolis algorithm for simulating the Ising model near T_c?"*
> - Agent:
>   - Thought: *"I should search our paper corpus for information on the Wolff cluster algorithm."*
>   - Action: `search_papers("Wolff cluster algorithm Ising model critical slowing down")` → retrieves passages from 2 papers
>   - Final Answer: *"According to [Wolff, 1989], the cluster algorithm updates entire clusters of aligned spins in a single step, rather than flipping individual spins as in Metropolis. This dramatically reduces critical slowing down near T_c. [Newman & Barkema, 1999] report that the autocorrelation time scales as τ ~ L^0.25 for Wolff vs τ ~ L^2.17 for Metropolis on an L×L lattice, making Wolff orders of magnitude faster for large systems near criticality."*
> - The agent now *cites real sources*, not hallucinated references.

#### After Module 1.3 (MCP Tools) — The agent can run Ising simulations and search arXiv

This is the transformative step. Via an MCP server, the agent gains two new tools: `search_arxiv` (live arXiv search) and `run_ising_simulation` (Monte Carlo simulation). The agent can now *do physics*, not just talk about it.

> **Example interaction:**
> - Student: *"Run an Ising simulation on a 32×32 lattice at T = 2.0, T = 2.269, and T = 3.0 with 10,000 Monte Carlo steps each. Describe what happens to the magnetization."*
> - Agent:
>   - Action: `run_ising_simulation(lattice_size=32, temperature=2.0, num_steps=10000, algorithm="wolff")`
>     → {magnetization_mean: 0.92, magnetization_std: 0.03, energy_mean: -1.75, ...}
>   - Action: `run_ising_simulation(lattice_size=32, temperature=2.269, num_steps=10000, algorithm="wolff")`
>     → {magnetization_mean: 0.41, magnetization_std: 0.18, energy_mean: -1.42, ...}
>   - Action: `run_ising_simulation(lattice_size=32, temperature=3.0, num_steps=10000, algorithm="wolff")`
>     → {magnetization_mean: 0.05, magnetization_std: 0.04, energy_mean: -1.10, ...}
>   - Final Answer: *"The results clearly show the phase transition:*
>     - *At T=2.0 (below T_c): the system is in the ordered phase with high magnetization |M|≈0.92. Spins are mostly aligned.*
>     - *At T=2.269 (≈T_c): the system is at criticality with intermediate magnetization |M|≈0.41 and large fluctuations (std=0.18). This is where long-range correlations develop.*
>     - *At T=3.0 (above T_c): the system is disordered with near-zero magnetization |M|≈0.05. Thermal fluctuations dominate."*
> - The agent is now an **experimentalist** that can design and run simulations.

#### After Module 1.4 (Reflection) — The agent self-checks and learns from mistakes

The agent now has a "Critic" that reviews its answers for errors, and a "Reflexion memory" that stores lessons learned across tasks.

> **Example interaction (showing error correction):**
> - Student: *"Estimate the critical temperature of the 2D Ising model by running a simulation on a 16×16 lattice."*
> - Agent (first attempt):
>   - Runs simulations at several temperatures, finds the specific heat peak at T≈2.35
>   - Initial answer: *"From the specific heat peak, I estimate T_c ≈ 2.35."*
> - Critic: *"The exact critical temperature is T_c = 2.269. The agent's estimate of 2.35 is 3.6% off. This discrepancy is likely due to finite-size effects on the small 16×16 lattice: the apparent T_c is shifted upward for finite systems. The agent should either (a) run on a larger lattice or (b) apply finite-size scaling corrections to extrapolate to L→∞."*
> - Agent (revised answer): *"My initial estimate of T_c ≈ 2.35 from a 16×16 lattice is shifted from the exact value (2.269) due to finite-size effects. For a more accurate estimate, I recommend running on 32×32 and 64×64 lattices and applying finite-size scaling."*
> - Reflexion memory stores: *"Lesson: small lattice sizes produce shifted T_c estimates. Always use L≥32 or apply finite-size scaling corrections."*

#### Module 1.5 — The complete Physics Research Assistant

All capabilities are assembled into one agent. It can handle complex, multi-step research tasks:

> **End-to-end demo task:**
> *"I'm starting a project on the 2D Ising model. Search our paper corpus for recommended simulation parameters, then run a quick simulation at three temperatures (below, at, and above T_c) on a 32×32 lattice. Report the magnetization and energy at each temperature, and check whether your results are consistent with the theoretical predictions."*
>
> The agent: (1) searches papers via RAG for recommended parameters, (2) extracts temperature values and lattice sizes, (3) runs 3 simulations via MCP, (4) compares results with the exact T_c using the calculator, (5) self-checks via reflection that the simulated magnetization near T_c is consistent with the known critical behavior.

**The single agent's architecture at the end of Day 1:**

```
┌──────────────────────────────────────────────────────────────┐
│              Physics Research Assistant (Day 1)              │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐    │
│   │          ReAct Loop (smolagents)                    │    │
│   │   Thought → Action → Observation → ... → Answer    │    │
│   └──────────────────────┬──────────────────────────────┘    │
│                          │                                   │
│          ┌───────────────┼───────────────────┐               │
│          │               │                   │               │
│   ┌──────▼──────┐ ┌──────▼──────┐  ┌────────▼────────┐      │
│   │ Local Tools │ │  RAG Tools  │  │   MCP Tools     │      │
│   │             │ │             │  │                  │      │
│   │ • calculator│ │ • search_   │  │ • search_arxiv   │      │
│   │ • unit_     │ │   papers    │  │ • run_ising_     │      │
│   │   converter │ │ (ChromaDB)  │  │   simulation     │      │
│   └─────────────┘ └─────────────┘  └──────────────────┘      │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐   │
│   │ Reflection (within-task) + Reflexion (across-task)   │   │
│   │ Critic prompt reviews answers; lessons stored in JSON│   │
│   └──────────────────────────────────────────────────────┘   │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐   │
│   │ LLM Backend: Ollama (qwen3.5:4b or :9b)             │   │
│   └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

**Why one agent is not enough (motivation for Day 2):**

The single agent works well for one question at a time, but struggles with a full systematic study because:
1. **Role confusion:** It mixes theoretical reasoning, simulation execution, and data analysis in one prompt — losing focus and making errors when tasks require deep specialization.
2. **Context overload:** A systematic study with 36 simulation runs (9 temperatures × 4 lattice sizes) produces too much data for one context window.
3. **No internal cross-checking:** There's no separation between the entity making predictions and the entity checking them — unlike a real research group where theory and experiment challenge each other.

---

### DAY 2 — The Multi-Agent System: "Physics Research Team"

On Day 2, we split the single agent into a team of 3 specialized agents that collaborate to conduct a **systematic study of the 2D Ising model phase transition**. Each agent has a distinct role, distinct tools, and distinct evaluation criteria — mirroring how actual physics research teams work.

#### The Three Agents

**Agent 1: The Theorist**
- **Role:** Senior Theoretical Physicist
- **What it does:** Derives predictions from first principles and literature. Produces a simulation plan for the team.
- **Tools:** `search_papers` (RAG), `calculator`
- **Produces:** A document containing:
  - Exact predictions: T_c = 2.269 J/k_B, β = 1/8, γ = 7/4, ν = 1
  - Finite-size scaling forms: M(L) ~ L^{-β/ν} f((T−T_c)·L^{1/ν})
  - A simulation plan: temperatures T = [1.5, 2.0, 2.1, 2.2, 2.269, 2.3, 2.4, 2.5, 3.0], lattice sizes L = [16, 32, 64], num_steps = 50,000, algorithm = Wolff
- **Quality criteria:** Mathematical rigor, correct citations, testable predictions

**Agent 2: The Experimentalist (Simulator)**
- **Role:** Senior Computational Physicist
- **What it does:** Takes the Theorist's simulation plan and executes it systematically. Reports raw results with uncertainties.
- **Tools:** `run_ising_simulation` (MCP), `search_arxiv` (MCP)
- **Produces:** A structured dataset containing:
  - Raw observables (M, E, C_v, χ) for every (T, L) pair requested by the Theorist
  - Statistical uncertainties (standard deviation, error of the mean)
  - Notes on any anomalies: *"L=16 runs near T_c show large autocorrelation times; consider discarding first 5,000 steps as thermalization."*
- **Quality criteria:** Systematic execution, proper error reporting, flagging of anomalies

**Agent 3: The Data Analyst**
- **Role:** Senior Data Scientist
- **What it does:** Processes the Experimentalist's raw data. Fits models, extracts critical exponents, produces visualizations.
- **Tools:** `python_executor` (runs Python/numpy/scipy/matplotlib in a sandbox)
- **Produces:**
  - Fitted critical exponents with confidence intervals: β = 0.131 ± 0.008 (exact: 0.125), γ = 1.72 ± 0.05 (exact: 1.75), ν = 0.98 ± 0.04 (exact: 1.0)
  - A comparison table: fitted vs exact values, with pass/fail assessment
  - A finite-size scaling collapse plot (M·L^{β/ν} vs (T−T_c)·L^{1/ν})
  - Assessment: *"All exponents are consistent with exact values within 1σ. The scaling collapse works well for L≥32 but shows deviations for L=16, likely due to corrections to scaling."*
- **Quality criteria:** Statistical rigor, proper error estimation, clear visualization

#### The Workflow

```
┌────────────────────────────────────────────────────────────────────────┐
│                     Physics Research Team (Day 2)                     │
│                                                                      │
│  TASK: "Conduct a systematic study of the 2D Ising model phase       │
│   transition. Derive predictions, run simulations, analyze results,  │
│   and produce a report comparing with exact solutions."              │
│                                                                      │
│  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐   │
│  │  THEORIST   │     │ EXPERIMENTALIST  │     │  DATA ANALYST   │   │
│  │             │     │                  │     │                 │   │
│  │ Derives:    │     │ Executes:        │     │ Analyzes:       │   │
│  │ • T_c       │────▶│ • 27 simulations │────▶│ • Fits β, γ, ν  │   │
│  │ • β, γ, ν   │     │   (9T × 3L)      │     │ • Scaling plot  │   │
│  │ • Scaling   │     │ • Reports raw    │     │ • Comparison    │   │
│  │   forms     │     │   data + errors  │     │   table         │   │
│  │ • Sim plan  │     │ • Flags issues   │     │ • Assessment    │   │
│  │             │     │                  │     │                 │   │
│  │ Tools:      │     │ Tools:           │     │ Tools:          │   │
│  │ • RAG       │     │ • MCP simulation │     │ • Python exec   │   │
│  │ • calculator│     │ • MCP arXiv      │     │   (numpy/scipy/ │   │
│  │             │     │                  │     │    matplotlib)  │   │
│  └─────────────┘     └──────────────────┘     └────────┬────────┘   │
│                                                        │            │
│                   ┌────────────────────────────────────┘            │
│                   │  REVIEW LOOP (Module 2.2)                       │
│                   ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  THEORIST (Review Task):                                    │    │
│  │  "Are the fitted exponents consistent with exact values?    │    │
│  │   If β=0.15±0.03, is this acceptable? Should we re-run?"   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Shared "Lab Notebook" (JSON): all agents read/write intermediate   │
│  results, enabling full traceability and cross-referencing.         │
└────────────────────────────────────────────────────────────────────────┘
```

#### Why 3 agents > 1 agent for this task

| Aspect | Single Agent (Day 1) | Multi-Agent Team (Day 2) |
|--------|---------------------|--------------------------|
| **Role clarity** | One prompt tries to be theorist + experimentalist + analyst | Each agent has a focused role, backstory, and evaluation criteria |
| **Context management** | 27 simulation results + theory + analysis overwhelm one context window | Each agent only sees what it needs: the Analyst doesn't see the full paper corpus |
| **Error detection** | Self-reflection catches some errors, but the same "brain" is checking itself | Cross-validation: the Theorist (independent from the Analyst) reviews the final results — like peer review |
| **Tool specialization** | All 5 tools in one agent → model often calls the wrong tool | Each agent has 2–3 tools → fewer choices, higher accuracy |
| **Scalability** | Adding more tasks means a longer prompt | Adding more agents means parallel work with clear interfaces |

#### Day 1 → Day 2 Component Mapping

Every Day 2 component is built from Day 1 building blocks. Nothing is thrown away.

| Day 1 Component | Reused in Day 2 as | By which agent |
|----------------|-------------------|----------------|
| `calculator` tool | Theorist's calculation tool | Theorist |
| `search_papers` (RAG + ChromaDB) | Theorist's literature search | Theorist |
| `search_arxiv` (MCP tool) | Experimentalist's literature search | Experimentalist |
| `run_ising_simulation` (MCP tool) | Experimentalist's primary tool | Experimentalist |
| MCP server (`physics_tools_server.py`) | Runs unchanged, serves both agents | Experimentalist |
| Reflection mechanism | Becomes the Theorist's "Review Task" (Module 2.2) | Theorist (review role) |
| Reflexion memory (JSON) | Becomes the shared "Lab Notebook" | All agents |
| smolagents agent | Wrapped as an A2A server for cross-system tasks (Module 2.3) | External delegation |

#### Concrete example of the multi-agent run

Here is what a successful run looks like (simplified):

**Step 1 — Theorist output:**
> *"For the 2D Ising model on a square lattice with nearest-neighbor interactions:*
> - *Exact critical temperature: T_c = 2/(ln(1+√2)) ≈ 2.269 J/k_B*
> - *Exact critical exponents: β = 1/8 = 0.125, γ = 7/4 = 1.75, ν = 1*
> - *Finite-size scaling: near T_c, the magnetization scales as M(T,L) = L^{-β/ν} · f((T−T_c) · L^{1/ν})*
> - *Simulation plan: temperatures [1.5, 2.0, 2.1, 2.2, 2.269, 2.3, 2.4, 2.5, 3.0], lattice sizes [16, 32, 64], 50,000 Wolff steps per run, discard first 10,000 for thermalization."*

**Step 2 — Experimentalist output:**
> *"Executed 27 simulations (9 temperatures × 3 lattice sizes). Summary:*
> | T | L=16: |M| | L=32: |M| | L=64: |M| |
> |---|---|---|---|
> | 1.5 | 0.985 ± 0.002 | 0.987 ± 0.001 | 0.988 ± 0.001 |
> | ... | ... | ... | ... |
> | 2.269 | 0.38 ± 0.12 | 0.27 ± 0.08 | 0.19 ± 0.05 |
> | ... | ... | ... | ... |
> | 3.0 | 0.06 ± 0.04 | 0.03 ± 0.02 | 0.02 ± 0.01 |
>
> *Note: L=16 runs near T_c show autocorrelation times ~200 steps; statistics may be marginal. L=32 and L=64 look clean."*

**Step 3 — Data Analyst output:**
> *"Fitted critical exponents using finite-size scaling:*
> - *β = 0.128 ± 0.009 (exact: 0.125) ✓*
> - *γ = 1.73 ± 0.06 (exact: 1.75) ✓*
> - *ν = 0.99 ± 0.04 (exact: 1.0) ✓*
>
> *All exponents are consistent with exact values within 1σ. Scaling collapse quality: χ²/dof = 1.2 for L=32,64; inclusion of L=16 worsens the fit (χ²/dof = 2.8), confirming the Experimentalist's concern about small-lattice statistics. Recommendation: exclude L=16 from final analysis or add L=128 runs."*

**Step 4 — Theorist review (Module 2.2):**
> *"The fitted exponents are all consistent with the exact Onsager solution. The Data Analyst's decision to exclude L=16 is well-justified by the poor scaling collapse. I recommend publishing with L=32 and L=64 data. One additional check: the Data Analyst should verify the universality class by confirming that the specific heat diverges logarithmically (α=0), not as a power law."*

---

## DAY 1: Building a Single AI Agent (4 hours)

### Narrative arc

Start with a naked LLM call. End with a fully functional Physics Research Assistant that can reason, search papers, run simulations, and self-correct. Each module adds exactly one capability.

---

### Module 1.0 — The Naked LLM (20 min)

**Concept:** What does an LLM actually do when you call it? Demystify the interface before building agents.

**Content:**
- Call Ollama's `/api/chat` endpoint directly from Python (raw `requests.post`, no library)
- Inspect the request/response: system prompt, messages list, temperature, token generation
- The LLM as a stateless function: `f(messages) → next_token_distribution`
- Key insight for physicists: the model samples from a Boltzmann-like distribution over the vocabulary, shaped by temperature. Lower T → greedy/deterministic, higher T → more entropic. The `temperature` parameter literally controls the softmax scaling, just like β = 1/kT in statistical mechanics.
- Multi-turn conversation: manually managing the messages list to maintain context

**Hands-on exercise:**
- Write a minimal `chat(messages, temperature) → str` function (~15 lines) that talks to Ollama
- Experiment: ask "What is the critical temperature of the 2D Ising model?" at temperature 0.1 vs 1.0 vs 2.0 — observe output variance and hallucination rate
- Experiment: multi-turn conversation where you first ask about T_c, then follow up with "How was this derived?"

**Notebook:** `01_naked_llm.ipynb`

**Outputs:** A working `chat()` function that will be reused in Module 1.1.

> **Key takeaway:** An LLM is a conditional text generator. Everything we build from here is scaffolding around this single operation.

---

### Module 1.1 — ReAct: Reasoning and Acting (40 min)

**Concept:** The ReAct paradigm — make the LLM alternate between *thinking* and *acting* in an explicit loop.

**Content:**
- The ReAct paper (Yao et al., 2023): Thought → Action → Observation → Thought → ...
- **Part A — Build ReAct from scratch (~20 min):**
  - System prompt that instructs the model to produce `Thought:`, `Action:`, `Observation:` blocks
  - Define 2 simple tools as Python functions:
    - `calculator(expression: str) → float` — evaluates math expressions safely (using `ast.literal_eval` or `simpleeval`)
    - `unit_converter(value: float, from_unit: str, to_unit: str) → float` — physics unit conversions (eV↔J, Kelvin↔eV, lattice units↔SI, etc.)
  - A Python `while` loop that: (1) calls the LLM, (2) parses the output for Action blocks, (3) executes the action, (4) feeds the Observation back, (5) terminates on `Final Answer:`
- **Part B — Introduce smolagents (~20 min):**
  - Show how smolagents implements the same loop in ~1000 lines
  - `ToolCallingAgent` (JSON-based tool calls) vs `CodeAgent` (writes Python code to call tools) — explain the difference, use `ToolCallingAgent` for this course
  - Connect to Ollama via `LiteLLMModel(model_id="ollama_chat/qwen3.5:4b")`
  - Wrap the calculator and unit converter as smolagents `@tool` functions

**Hands-on exercise:**
- Part A: Run the from-scratch ReAct loop:
  - Task: *"What is the Schwarzschild radius of a black hole with mass equal to 10 solar masses? Express the answer in kilometers."*
  - The agent should think about the formula (r_s = 2GM/c²), call the calculator, call the unit converter
- Part B: Rewrite the same task using smolagents — observe how the framework handles parsing, error recovery, and retries automatically
- Bonus task: *"The critical temperature of the 2D Ising model is T_c = 2J/ln(1+√2). If the coupling constant J = 0.5 eV, what is T_c in Kelvin?"*

**Notebook:** `02_react_agent.ipynb`

**Outputs:** A working smolagents `ToolCallingAgent` with calculator + unit converter tools, connected to Ollama.

> **Key takeaway:** An agent = an LLM + a loop + tools. The ReAct pattern is the minimal viable agent architecture.

---

### Module 1.2 — RAG: Retrieval-Augmented Generation (50 min)

**Concept:** Give the agent access to a corpus of physics papers so it can ground its answers in real literature.

**Content:**
- Why RAG? Context windows are finite, LLMs hallucinate references, and knowledge has a cutoff date
- The RAG pipeline, step by step:
  1. **Load:** Read PDFs from `data/papers/` (using `pypdf` or `pdfplumber`)
  2. **Chunk:** Split each paper into passages of ~300–500 tokens (with overlap)
  3. **Embed:** Use `sentence-transformers` (`all-MiniLM-L6-v2`) to map each chunk to a 384-dim vector
  4. **Store:** Index vectors in ChromaDB (local, serverless, no setup)
  5. **Retrieve:** Given a query, embed it and find top-K similar chunks (cosine similarity)
  6. **Generate:** Inject retrieved chunks into the LLM prompt as context
- Build each step in plain Python, then wrap the retrieval+generation as a smolagents tool:
  - `search_papers(query: str, top_k: int = 3) → str` — returns the top-K most relevant passages with paper titles and page numbers
- Discuss: chunking strategies (fixed-size vs semantic), embedding model choice, retrieval quality (what if the right answer spans two chunks?), the trade-off between retrieving too much (noise) vs too little (missing information)
- For physicists: embedding space as a high-dimensional metric space; retrieval as approximate nearest-neighbor search

**Hands-on exercise:**
- Ingest the pre-downloaded paper corpus (5–10 Ising model papers)
- Build the retrieval pipeline from scratch, test it standalone with a query
- Wrap as a smolagents tool, add to the agent from Module 1.1
- Tasks:
  - *"What numerical methods have been used to estimate critical exponents in the 2D Ising model? Cite specific papers."*
  - *"What is the Wolff algorithm and how does it compare to the Metropolis algorithm for simulating the Ising model near T_c?"*
  - *"Do any papers in our corpus discuss the Ising model on non-regular lattices?"*
- The agent should retrieve relevant passages, synthesize them, and cite the source papers

**Notebook:** `03_rag_agent.ipynb`

**Outputs:** A ChromaDB vector store loaded with the paper corpus; a `search_papers` tool added to the agent.

> **Key takeaway:** RAG extends the agent's knowledge beyond training data. The retrieval step is a learned similarity search — familiar to anyone who has worked with metric spaces.

---

### ☕ Break (15 min)

---

### Module 1.3 — Tool Use & MCP: The Model Context Protocol (50 min)

**Concept:** Standardize how agents discover and use external tools via MCP — and give the agent the ability to *run physics simulations*.

**Content:**
- **What is MCP?** The emerging standard for connecting LLMs to tools and data sources. Think of it as a universal API adapter: instead of hardcoding each tool, the agent discovers available tools at runtime from any MCP server.
- **MCP architecture:**
  - Client (the agent) ↔ Server (exposes tools) via JSON-RPC
  - Transport: stdio (local) or HTTP+SSE (remote)
  - Key operations: `tools/list` (discover what tools exist), `tools/call` (execute a tool with arguments)
  - Tool schema: each tool has a name, description, and JSON Schema for its inputs
- **Build an MCP server** (Python, using FastAPI or the `mcp` SDK) that exposes 2 physics-relevant tools:
  1. **`search_arxiv(query: str, max_results: int) → list[dict]`**
     - Searches the arXiv API live
     - Returns: list of {title, authors, abstract, arxiv_id, url}
     - Use case: finding papers beyond the local RAG corpus
  2. **`run_ising_simulation(lattice_size: int, temperature: float, num_steps: int, algorithm: str) → dict`**
     - Runs a Monte Carlo simulation of the 2D Ising model (Metropolis or Wolff algorithm)
     - Code is pre-written in the MCP server (using numpy) — the agent doesn't write the simulation, it *calls* it
     - Returns: {magnetization_mean, magnetization_std, energy_mean, energy_std, specific_heat, susceptibility, final_configuration}
     - This simulates the idea of an agent controlling lab equipment / running experiments
- **Connect MCP to smolagents:**
  - smolagents has native MCP support via `ToolCollection.from_mcp()`
  - The agent dynamically discovers the tools from the server at startup
  - Demo: the agent now has calculator + unit_converter + search_papers (RAG) + search_arxiv (MCP) + run_ising_simulation (MCP)
- **Discussion:** MCP vs hardcoded tools — when to use which. The composability advantage: anyone can publish an MCP server, any agent can consume it. Brief tour of the MCP ecosystem.

**Hands-on exercise:**
- Start the pre-built MCP server (`mcp_server/physics_tools_server.py`)
- Connect it to the smolagents agent and verify tool discovery (`tools/list`)
- Tasks:
  - *"Search arXiv for the latest papers on machine learning approaches to the Ising model. Summarize the top 3 results."*
  - *"Run an Ising simulation on a 32×32 lattice at T = 2.0, T = 2.269 (T_c), and T = 3.0 with 10000 Monte Carlo steps each. Describe what happens to the magnetization across these temperatures."*
  - The agent should observe: ordered phase (high |M|) at T=2.0, critical fluctuations at T_c, disordered phase (M≈0) at T=3.0
- Students modify the MCP server to add one custom tool (e.g., `compute_correlation_length`, `measure_binder_cumulant`, or any physics observable they find useful)

**Notebook:** `04_mcp_tools.ipynb`
**Support file:** `mcp_server/physics_tools_server.py`
**Support file:** `mcp_server/ising_simulator.py` (the actual Monte Carlo code, imported by the server)

**Outputs:** A running MCP server with arXiv + Ising simulation tools; the agent now has 5 tools total.

> **Key takeaway:** MCP decouples tool implementation from agent logic. The Ising simulation could be replaced with a real instrument API — the agent code wouldn't change.

---

### Module 1.4 — Reflection & Reflexion: Self-Improving Agents (40 min)

**Concept:** Make the agent evaluate its own outputs and learn from past mistakes.

**Content:**
- **Reflection (within-task self-correction):**
  - After the agent produces a `Final Answer`, add a second LLM call with a "Critic" system prompt:
    *"You are a senior physicist reviewing a junior colleague's work. Check for: (1) mathematical correctness, (2) consistency between theoretical predictions and simulation results, (3) proper citation of sources, (4) any signs of hallucination. Be specific about what's wrong and why."*
  - If the critic identifies issues, the agent loops back and revises (max 2 retries)
  - Implement this as a wrapper around the smolagents agent
- **Reflexion (across-task learning, Shinn et al. 2023):**
  - After completing a task, the agent stores a `lesson_learned` in a JSON file:
    - What went well, what went wrong, what to do differently next time
  - On future tasks, past lessons are injected into the system prompt
  - Implement a simple `ReflexionMemory` class that reads/writes a JSON file
- **Demo: catching a finite-size effect error:**
  - Ask the agent to estimate T_c from a simulation on a 8×8 lattice
  - The simulation will give T_c ≈ 2.4 (shifted from the exact 2.269 due to finite-size effects)
  - Without reflection: the agent reports T_c ≈ 2.4 confidently
  - With reflection: the critic catches the discrepancy with the exact solution and suggests running on a larger lattice or applying finite-size scaling corrections
- **For physicists:** Reflection ≈ iterative numerical methods (solve, check residual, correct). Reflexion ≈ experimental logbook (record what worked for next time).

**Hands-on exercise:**
- Add reflection to the agent — run a task where it initially makes an error
- Add Reflexion memory — run a sequence of 3 tasks and show improvement:
  1. *"Estimate the critical temperature of the 2D Ising model by running a simulation on a 16×16 lattice."* (Agent gets ≈2.35, critic flags finite-size effects)
  2. *"Now estimate T_c using a 64×64 lattice."* (Agent remembers the lesson, gets closer to 2.269)
  3. *"Compare the critical exponents β for the 2D Ising model: theoretical prediction vs Monte Carlo estimate from your simulations."* (Agent knows to use larger lattices from lesson learned)
- Examine the reflection memory file — what lessons did the agent learn?

**Notebook:** `05_reflection_agent.ipynb`

**Outputs:** A `ReflexionMemory` class; the agent now self-corrects and learns across tasks.

> **Key takeaway:** Reflection turns a one-shot generator into an iterative reasoner. Reflexion adds learning across tasks. Together, they're the minimal ingredients for a self-improving agent.

---

### Module 1.5 — The Complete Physics Research Assistant (25 min)

**Concept:** Assemble everything into a single coherent agent. Recap the architecture. Preview Day 2.

**Content:**
- Architecture recap (build the diagram live):
  ```
  ┌──────────────────────────────────────────────────┐
  │          Physics Research Assistant               │
  │                                                  │
  │  LLM (Ollama/qwen3.5) ← ReAct loop (smolagents) │
  │       │                                          │
  │       ├── calculator          (local tool)       │
  │       ├── unit_converter      (local tool)       │
  │       ├── search_papers       (RAG/ChromaDB)     │
  │       ├── search_arxiv        (MCP tool)         │
  │       ├── run_ising_simulation(MCP tool)         │
  │       │                                          │
  │       └── Reflection + Reflexion memory          │
  └──────────────────────────────────────────────────┘
  ```
- **End-to-end demo task:**
  *"I'm starting a project on the 2D Ising model. Search our paper corpus for recommended simulation parameters, then run a quick simulation at three temperatures (below, at, and above T_c) on a 32×32 lattice. Report the magnetization and energy at each temperature, and check whether your results are consistent with the theoretical predictions."*
- The agent should: (1) search papers via RAG, (2) extract recommended parameters, (3) run 3 simulations via MCP, (4) compare with theory using the calculator, (5) self-check via reflection
- **Motivation for Day 2:** This works, but it's one agent doing everything. What if the task is more complex — a full systematic study with 36 simulation runs, careful statistical analysis, and publication-quality plots? A single agent loses track, confuses roles, and runs out of context. Tomorrow we split it into a team.

**Hands-on exercise:**
- Students run the end-to-end task with their fully assembled agent
- Examine the full trace: every thought, action, observation, and reflection
- Identify failure modes: where does the single agent struggle?

**Notebook:** `06_full_agent.ipynb`

**Outputs:** The complete Physics Research Assistant — this is the building block for Day 2.

> **Key takeaway:** A modern AI agent is five things: an LLM, a reasoning loop, knowledge retrieval (RAG), external actions (MCP tools), and self-correction (reflection). Nothing more, nothing less.

---

## DAY 2: Building a Multi-Agent System (4 hours)

### Narrative arc

Start from the single agent built on Day 1. Discover *why* multiple agents beat one generalist. Build a 3-agent physics research team with CrewAI, add inter-agent communication patterns, then connect to the broader ecosystem via the A2A protocol. Close with the instructor's research on Ising LLMs and Naming Game LLMs.

---

### Module 2.0 — Why Multi-Agent? Motivation & Architecture Patterns (30 min)

**Concept:** When a single agent is not enough, and how to decompose a problem into collaborating agents.

**Content:**
- **The failure of the single agent:** revisit the end of Day 1. The Physics Research Assistant can handle one question at a time, but a full systematic study requires:
  - Theoretical expertise (knowing which predictions to test, which scaling forms to use)
  - Experimental expertise (designing a simulation campaign: which temperatures, lattice sizes, how many MC steps, which algorithm)
  - Data analysis expertise (fitting, error estimation, scaling collapse, visualization)
  - A single agent conflates these roles, loses track in long contexts, and can't self-specialize
- **Multi-agent design patterns** (with diagrams):
  1. **Pipeline / Sequential:** Agent A → Agent B → Agent C (each processes the previous output)
  2. **Hierarchical / Orchestrator:** A manager agent delegates sub-tasks to specialists
  3. **Debate / Adversarial:** Agents argue, a judge resolves
  4. **Swarm / Peer-to-peer:** Agents self-organize
- **Mapping to physics:**
  - Pipeline ≈ data processing chain in experimental physics (trigger → reconstruction → analysis)
  - Hierarchical ≈ PI + postdocs + students in a research group
  - Debate ≈ theory vs experiment confrontation (crucial for catching systematic errors)
  - Swarm ≈ emergent behavior in many-body systems
- **Introduce CrewAI:**
  - Why CrewAI: simpler than AG2/AutoGen (which is in maintenance mode), role-based metaphor maps to our physics team, Ollama-compatible, actively maintained
  - Core abstractions: **Agent** (role + goal + backstory + tools), **Task** (description + expected_output + agent), **Crew** (agents + tasks + process type)
  - Process types: `sequential` (pipeline), `hierarchical` (manager delegates)

**Notebook:** `07_multi_agent_intro.ipynb` (conceptual diagrams + first CrewAI hello-world)

> **Key takeaway:** Multi-agent design is about decomposing a problem into roles with different expertise, tools, and evaluation criteria — mirroring how real research teams work.

---

### Module 2.1 — The Physics Research Team: 3 Agents with CrewAI (60 min)

**Concept:** Build a 3-agent team that collaborates to conduct a systematic study of the 2D Ising model phase transition.

**Content:**
- **Define three agents in CrewAI:**

  **Agent 1 — The Theorist**
  - Role: `"Senior Theoretical Physicist"`
  - Goal: `"Derive theoretical predictions for the 2D Ising model phase transition and design a rigorous simulation plan"`
  - Backstory: `"You are an expert in statistical mechanics with deep knowledge of exactly solvable models. You always ground your work in the literature and provide precise mathematical predictions that experimentalists can test."`
  - Tools: `search_papers` (RAG from Day 1), `calculator`
  - Expected output: A document containing (1) exact predictions for T_c, β, γ, ν, (2) finite-size scaling forms, (3) a simulation plan (list of temperatures, lattice sizes, MC steps)

  **Agent 2 — The Experimentalist**
  - Role: `"Senior Computational Physicist / Experimentalist"`
  - Goal: `"Execute the simulation campaign designed by the Theorist and report raw results with uncertainty estimates"`
  - Backstory: `"You are an expert in Monte Carlo simulations. You care about systematic errors, thermalization, autocorrelation times, and statistical quality. You execute simulations methodically and flag any anomalies."`
  - Tools: `run_ising_simulation` (MCP from Day 1), `search_arxiv` (MCP from Day 1)
  - Expected output: Raw simulation data (magnetization, energy, specific heat, susceptibility) for all requested (T, L) pairs, with error bars and notes on any issues

  **Agent 3 — The Data Analyst**
  - Role: `"Senior Data Scientist"`
  - Goal: `"Analyze the simulation results, extract critical exponents via finite-size scaling, and produce a comparison with exact theoretical predictions"`
  - Backstory: `"You are an expert in statistical data analysis and scientific visualization. You fit power laws, perform scaling collapses, estimate confidence intervals, and produce publication-quality figures. You are skeptical of results that don't pass statistical tests."`
  - Tools: `python_executor` (a sandboxed tool that runs Python/numpy/scipy/matplotlib code and returns text output + saved figures)
  - Expected output: (1) Fitted values of β, γ, ν with confidence intervals, (2) comparison table: fitted vs exact, (3) finite-size scaling collapse plot, (4) overall assessment of data quality

- **Define the tasks and crew:**
  - Task 1 (Theorist): *"Derive the theoretical predictions for the 2D Ising model phase transition. Specify the exact critical temperature, critical exponents, and finite-size scaling forms. Then design a simulation plan: recommend specific temperatures (at least 8 values spanning the transition), lattice sizes (at least 3, covering L=16 to L=64), and the number of Monte Carlo steps."*
  - Task 2 (Experimentalist): *"Execute the simulation plan from the Theorist. For each (temperature, lattice_size) pair, run the Ising simulation using the Wolff algorithm. Record magnetization, energy, specific heat, and magnetic susceptibility. Report raw data with statistical uncertainties. Flag any runs where thermalization appears insufficient."*
  - Task 3 (Data Analyst): *"Analyze the simulation data from the Experimentalist. Extract the critical exponents β, γ, and ν using finite-size scaling techniques. Perform a scaling collapse of the magnetization data. Compare your fitted exponents with the exact theoretical values from the Theorist. Produce a summary table and assessment of the study."*
  - Crew process: `sequential` (Theorist → Experimentalist → Data Analyst)

- **Run the crew** and examine the output:
  - Each agent's full output is displayed
  - The conversation transcript shows how context flows between agents
  - Discuss: where did the pipeline work well? Where did information get lost?

**Hands-on exercise:**
- Students implement the 3-agent crew from scratch using CrewAI
- Connect CrewAI agents to Ollama via LiteLLM
- Plug in the tools from Day 1 (RAG, MCP)
- Run the crew on the Ising model systematic study task
- Experiment: modify agent backstories and observe behavior changes (e.g., make the Data Analyst overly conservative — it rejects results that should be accepted)
- Experiment: switch from `sequential` to `hierarchical` process (add a Manager agent that decides task order) — compare results and efficiency

**Notebook:** `08_physics_crew.ipynb`

**Outputs:** A working 3-agent CrewAI physics team that reuses Day 1 infrastructure.

> **Key takeaway:** A multi-agent system's quality depends on three things: role design (system prompts), task decomposition (what each agent is asked to do), and coordination process (how outputs flow between agents).

---

### ☕ Break (15 min)

---

### Module 2.2 — Agent Communication, Memory & Cross-Validation (45 min)

**Concept:** Make agents share knowledge, maintain shared state, and check each other's work.

**Content:**
- **Communication patterns in CrewAI:**
  - Sequential: output of Task N becomes context for Task N+1 (what we just did)
  - Hierarchical: a Manager agent reads the task descriptions, decides who works on what, and synthesizes
  - Custom callbacks: intercept messages between agents for logging, filtering, or transformation
- **Shared memory — the "Lab Notebook":**
  - Problem: in the sequential process, the Data Analyst only sees the Experimentalist's output, not the Theorist's full derivation. Information degrades along the pipeline.
  - Solution: implement a shared memory that all agents can read and write — like a lab notebook
  - In CrewAI: use the `memory` parameter (short-term, long-term, entity memory)
  - Also implement a custom shared state: a JSON "lab notebook" file that agents append to and read from
- **Cross-validation — the review loop:**
  - After the Data Analyst produces results, the Theorist reviews the report
  - Add a 4th task: *"Review the Data Analyst's conclusions. Check whether the fitted critical exponents are consistent with the exact values within the reported uncertainties. If not, identify the likely source of error (finite-size effects, insufficient thermalization, fitting methodology) and recommend specific corrections."*
  - This creates a pipeline with a feedback loop: Theorist → Experimentalist → Analyst → Theorist (review)
  - If the review finds issues, the crew can iterate (up to a maximum number of rounds)
- **Handling disagreements:**
  - Demo: introduce a deliberate error (e.g., the Experimentalist uses too few MC steps, producing noisy data)
  - The Data Analyst reports poor fits with large error bars
  - The Theorist's review catches the inconsistency and recommends more MC steps
  - Discuss: this is exactly how real research groups work — cross-checking between theory and experiment
- **Human-in-the-loop:**
  - Add a checkpoint where the crew pauses and asks the user: "The Theorist and Data Analyst disagree on whether the β estimate is acceptable. The fitted value is 0.15 ± 0.03, the exact value is 0.125. Should we (a) accept with caveat, (b) re-run with larger lattice, (c) change fitting method?"

**Hands-on exercise:**
- Add shared memory (lab notebook) to the crew
- Add the Theorist review task as a 4th step
- Introduce a deliberate error and observe the review loop catching it
- Add a human-in-the-loop checkpoint
- Task: *"The simulation data shows an unexpected bump in the specific heat curve at T = 1.8, well below T_c. The team should diagnose the anomaly: the Theorist checks if it's physically plausible, the Experimentalist re-runs with different parameters, and the Data Analyst compares before/after."*

**Notebook:** `09_agent_communication.ipynb`

**Outputs:** An enhanced crew with shared memory, cross-validation, and human-in-the-loop capabilities.

> **Key takeaway:** Multi-agent reliability comes from cross-validation between agents with different expertise — the same principle that makes peer review work in science.

---

### Module 2.3 — The A2A Protocol: Agent-to-Agent Interoperability (45 min)

**Concept:** How agents from *different* systems discover and talk to each other via the A2A protocol.

**Content:**
- **The problem A2A solves:**
  - Your physics team lives in CrewAI. A colleague has built a data analysis agent in LangGraph. Another group has a literature search agent using smolagents. How do they collaborate?
  - Today: they can't, because each framework has its own internal messaging format
  - A2A: a universal protocol for inter-agent communication, regardless of framework
- **A2A vs MCP — the two protocols of the agentic stack:**
  - **MCP** = agent ↔ tool (how an agent accesses a capability, like a simulation or database)
  - **A2A** = agent ↔ agent (how two agents communicate as peers, exchanging tasks and results)
  - Analogy: MCP is like USB (connecting peripherals), A2A is like HTTP (connecting services)
- **A2A core concepts:**
  1. **Agent Card** — JSON at `/.well-known/agent-card.json`: name, description, version, skills, endpoint URL, authentication. Like a business card + CV for an agent.
  2. **Tasks** — the work unit. Client agent sends a task → server agent processes it → returns result. Lifecycle: `submitted → working → completed/failed`.
  3. **Messages & Artifacts** — agents exchange messages (text, structured data) and produce artifacts (files, data, results).
  4. **Streaming & push notifications** — for long-running tasks (like a simulation that takes minutes).
- **Build a minimal A2A setup:**
  - **A2A server:** Wrap the Day 1 Physics Research Assistant as an A2A-compliant server using FastAPI:
    - Endpoint: `POST /tasks` — accepts a task, runs the agent, returns results
    - Agent card: `GET /.well-known/agent-card.json` — describes the agent's capabilities
    - Skills: "literature_search", "ising_simulation", "physics_calculation"
  - **A2A client:** A simple Python client that:
    - Discovers the server by fetching its agent card
    - Sends a task and waits for the result
  - **Cross-system demo:** The CrewAI physics team's Data Analyst needs to run an additional simulation but the MCP server is busy. Instead, it delegates to an external A2A agent (the Day 1 agent, now wrapped as an A2A server) — different framework, same protocol.
- **Discussion:** The vision of an ecosystem of interoperable agents. Agents as microservices. The Linux Foundation stewardship of A2A. The parallels with how HTTP enabled the web.

**Hands-on exercise:**
- Students wrap their Day 1 agent as an A2A server (starter code provided in `a2a_server/`)
- Test locally: the A2A client sends a task, the server processes it, returns the result
- **Pair exercise:** Exchange agent card URLs with a neighbor. Student A's CrewAI team delegates a task to Student B's A2A agent and vice versa
- Task: *"Your Data Analyst needs the correlation function C(r) for the Ising model at T_c, but your MCP server doesn't have that tool. Delegate to your neighbor's A2A agent, which has implemented it as a custom MCP tool yesterday."*
- Examine the A2A messages: task submission, status updates, artifact delivery

**Notebook:** `10_a2a_protocol.ipynb`
**Support file:** `a2a_server/agent_server.py` (FastAPI A2A wrapper)
**Support file:** `a2a_server/agent_card.json` (template agent card)

**Outputs:** An A2A-compliant wrapper for the Day 1 agent; experience with cross-system agent communication.

> **Key takeaway:** A2A makes agents composable across organizational and framework boundaries. If MCP is the USB-C for tools, A2A is the HTTP for agents.

---

### Module 2.4 — Scaling, Autoresearch & Open Frontiers (30 min)

**Concept:** What happens at scale? Case studies and research frontiers.

**Content:**
- **Scaling challenges:**
  - Communication complexity: N agents can mean O(N²) messages
  - The "too many cooks" problem: adding agents to simple tasks degrades performance
  - Hierarchical organization as a scaling strategy (managers + teams)
  - Context window management across many agents
- **Case study: Karpathy's autoresearch**
  - The pattern: an AI agent autonomously modifies training code → runs a 5-min experiment → evaluates against a metric → keeps or reverts via git → repeats
  - The key innovation: `program.md` — the human writes research directions in natural language, the agent executes. This is a new programming paradigm.
  - Results: 700 experiments overnight, 20 genuine improvements, 11% speedup on already-optimized code
  - Karpathy's vision: *"The goal is not to emulate a single PhD student, it's to emulate a research community of them"* — connecting to multi-agent scaling
  - **For the audience:** imagine autoresearch applied to your field. What metric would you optimize? What would your `program.md` say?
- **Research frontiers — where physics meets multi-agent AI:**
  - Can we derive scaling laws for multi-agent systems? (Analogous to neural scaling laws)
  - Phase transitions in multi-agent consensus: when N agents debate, under what conditions do they converge vs. polarize?
  - Is there a "critical temperature" for multi-agent deliberation? (Temperature parameter as a control variable for agent diversity)
  - Open question: what is the optimal number of agents for a given task complexity? Is there a phase diagram?
  - Preview of the instructor's research: these questions can be studied rigorously using tools from statistical mechanics

**Notebook:** `11_scaling_frontiers.ipynb` (conceptual + small demo)

> **Key takeaway:** Multi-agent systems are not just an engineering tool — they are a new kind of physical system whose collective behavior can be studied with statistical mechanics.

---

### Module 2.5 — Capstone: Ising LLMs & Naming Game LLMs (30 min)

**Concept:** The instructor's research — studying multi-agent LLM dynamics as physical systems.

**Content:**
- **Ising LLMs:** Map the agreement/disagreement dynamics of N LLM agents to spin interactions in the Ising model
  - Each agent has an "opinion" (spin ±1) on a topic
  - Agents interact pairwise: they exchange arguments and may flip their opinion
  - The "coupling constant" J is determined by argument quality / persuasiveness
  - Question: does the system exhibit a phase transition between consensus (ordered) and disagreement (disordered)?
  - Show real simulation results
- **Naming Game LLMs:** Emergent convention formation in LLM agent populations
  - N agents must agree on a name/label for a concept through pairwise interactions
  - No central authority — conventions emerge from local interactions
  - Classical Naming Game on networks is well-studied; what changes when agents are LLMs?
  - Show results: convergence dynamics, dependence on network topology, role of model diversity
- **Discussion:**
  - How do these results connect to what we built in the course?
  - The agents from Day 2 Module 2.1 are a simple instance of interacting LLM agents
  - What other physics models could describe multi-agent dynamics? (Voter model, Potts model, XY model...)
  - Open research directions for the audience

**Notebook:** `12_ising_naming_game_llms.ipynb` (instructor demo + discussion prompts)

> **Key takeaway:** The tools of statistical physics — phase transitions, order parameters, scaling laws, mean-field theory — are the right language for understanding the collective behavior of multi-agent LLM systems. This is an open research frontier where physicists have an unfair advantage.

---

## Complete Deliverables Table

All files to be built for the course repository:

### Setup & Infrastructure

| File | Type | Description |
|------|------|-------------|
| `setup/requirements.txt` | Config | All Python dependencies with pinned versions |
| `setup/setup_env.sh` | Script | Creates venv, installs packages, pulls Ollama model |
| `setup/download_papers.py` | Script | Downloads 5–10 arXiv papers on the 2D Ising model |
| `setup/verify_setup.py` | Script | Checks Ollama, model, packages, paper corpus |
| `mcp_server/physics_tools_server.py` | Script | MCP server exposing `search_arxiv` + `run_ising_simulation` |
| `mcp_server/ising_simulator.py` | Module | 2D Ising Monte Carlo code (Metropolis + Wolff algorithms) |
| `a2a_server/agent_server.py` | Script | A2A-compliant FastAPI wrapper for the Day 1 agent |
| `a2a_server/agent_card.json` | Config | Template A2A agent card |

### Day 1 Notebooks

| File | Module | Description |
|------|--------|-------------|
| `notebooks/01_naked_llm.ipynb` | 1.0 | Raw Ollama API calls, temperature experiments |
| `notebooks/02_react_agent.ipynb` | 1.1 | ReAct from scratch → smolagents, calculator + unit converter |
| `notebooks/03_rag_agent.ipynb` | 1.2 | Full RAG pipeline (PDF → chunks → embeddings → ChromaDB → tool) |
| `notebooks/04_mcp_tools.ipynb` | 1.3 | MCP server + client, arXiv search + Ising simulation |
| `notebooks/05_reflection_agent.ipynb` | 1.4 | Reflection critic + Reflexion memory, finite-size effect demo |
| `notebooks/06_full_agent.ipynb` | 1.5 | Complete Physics Research Assistant, end-to-end demo |

### Day 2 Notebooks

| File | Module | Description |
|------|--------|-------------|
| `notebooks/07_multi_agent_intro.ipynb` | 2.0 | Architecture patterns + first CrewAI hello-world |
| `notebooks/08_physics_crew.ipynb` | 2.1 | 3-agent physics team (Theorist + Experimentalist + Analyst) |
| `notebooks/09_agent_communication.ipynb` | 2.2 | Shared memory, review loop, human-in-the-loop |
| `notebooks/10_a2a_protocol.ipynb` | 2.3 | A2A server + client, cross-system delegation |
| `notebooks/11_scaling_frontiers.ipynb` | 2.4 | Autoresearch case study, scaling challenges |
| `notebooks/12_ising_naming_game_llms.ipynb` | 2.5 | Instructor's research demo (Ising LLMs, Naming Game) |

### Slides (Optional)

| File | Description |
|------|-------------|
| `slides/day1_slides.pdf` | Minimal lecture slides for Day 1 (mostly architecture diagrams) |
| `slides/day2_slides.pdf` | Minimal lecture slides for Day 2 (MAS patterns, A2A diagrams) |

---

## Build Order

Recommended order for developing the materials (dependencies shown):

```
Phase 1 — Infrastructure (build first, everything depends on it):
  setup/requirements.txt
  setup/setup_env.sh
  setup/download_papers.py
  setup/verify_setup.py
  mcp_server/ising_simulator.py       ← the Monte Carlo code, tested standalone
  mcp_server/physics_tools_server.py   ← depends on ising_simulator.py

Phase 2 — Day 1 Notebooks (build in order, each depends on previous):
  notebooks/01_naked_llm.ipynb
  notebooks/02_react_agent.ipynb       ← depends on 01 (chat function)
  notebooks/03_rag_agent.ipynb         ← depends on 02 (agent), setup/download_papers.py
  notebooks/04_mcp_tools.ipynb         ← depends on 03 (agent), mcp_server/
  notebooks/05_reflection_agent.ipynb  ← depends on 04 (full-tool agent)
  notebooks/06_full_agent.ipynb        ← depends on 05 (all components)

Phase 3 — Day 2 Notebooks (build in order):
  notebooks/07_multi_agent_intro.ipynb ← depends on Day 1 concepts
  notebooks/08_physics_crew.ipynb      ← depends on 06 (reuses tools), CrewAI setup
  notebooks/09_agent_communication.ipynb ← depends on 08 (crew)
  a2a_server/agent_server.py           ← depends on 06 (wraps the Day 1 agent)
  a2a_server/agent_card.json
  notebooks/10_a2a_protocol.ipynb      ← depends on a2a_server/, 08
  notebooks/11_scaling_frontiers.ipynb ← conceptual, light dependencies
  notebooks/12_ising_naming_game_llms.ipynb ← instructor's research, mostly demo

Phase 4 — Slides (build last, after notebooks stabilize):
  slides/day1_slides.pdf
  slides/day2_slides.pdf
```

---

## References & Resources

### Key Papers
- Yao et al. (2023) — "ReAct: Synergizing Reasoning and Acting in Language Models"
- Shinn et al. (2023) — "Reflexion: Language Agents with Verbal Reinforcement Learning"
- Lewis et al. (2020) — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Wu et al. (2023) — "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"
- Onsager (1944) — "Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition"

### Repositories
- [smolagents](https://github.com/huggingface/smolagents) — HuggingFace's minimal agent library (~1000 lines of core logic)
- [CrewAI](https://github.com/crewAIInc/crewAI) — Role-based multi-agent framework
- [rasbt/mini-coding-agent](https://github.com/rasbt/mini-coding-agent) — Minimal agent harness (reference architecture for Module 1.1)
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — Autonomous ML research agent (case study for Module 2.4)
- [A2A Protocol](https://github.com/a2aproject/A2A) — Agent-to-Agent open protocol (Linux Foundation)
- [MCP Specification](https://modelcontextprotocol.io/) — Model Context Protocol by Anthropic

### Documentation
- [smolagents docs](https://huggingface.co/docs/smolagents/index) — Guides, tutorials, API reference
- [CrewAI docs](https://docs.crewai.com/) — Agent/Task/Crew configuration
- [A2A Protocol docs](https://a2a-protocol.org/latest/) — Full specification + quickstart
- [Ollama API docs](https://ollama.com/docs/api) — Chat and generate endpoints
- [ChromaDB docs](https://docs.trychroma.com/) — Vector store for RAG

---

## Instructor Notes

### Timing risks
- **Module 1.2 (RAG)** and **Module 2.1 (first crew)** are the most likely to run long. Mitigation: provide working starter code so students who fall behind can load the pre-built version and continue.
- **Module 2.3 (A2A)** depends on networking between student laptops. Mitigation: have a fallback where students run both the A2A server and client on localhost (no cross-machine communication needed).

### Model recommendations
- `qwen3.5:4b` — minimum viable model. Runs on any modern laptop. Follows tool-calling instructions reasonably well.
- `qwen3.5:9b` — recommended if ≥16 GB RAM. Significantly better at structured output and multi-step reasoning.
- `llama3.1:8b` — alternative. Good at code generation (useful for the Data Analyst agent on Day 2).
- The entire course is model-agnostic. Swapping models requires changing one string.

### The Ising model simulation
- The Monte Carlo code in `mcp_server/ising_simulator.py` should be simple, correct, and fast enough for classroom use.
- Metropolis algorithm: straightforward, ~50 lines of numpy. Good for small lattices.
- Wolff cluster algorithm: faster near T_c, ~80 lines. Recommended for larger lattices.
- Target: a 32×32 lattice with 10,000 MC steps should complete in <5 seconds on any laptop.
- The simulation returns pre-computed observables (M, E, C_v, χ) so the agent doesn't need to compute them.

### Day 1 → Day 2 continuity
- The MCP server from Day 1 runs unchanged on Day 2 — the CrewAI Experimentalist agent calls the same tools.
- The RAG pipeline from Day 1 runs unchanged — the CrewAI Theorist agent queries the same ChromaDB.
- The reflection mechanism from Day 1 conceptually becomes the Theorist's review step on Day 2.
- Students should keep their Day 1 virtual environment and servers running. Day 2 builds on top, never replaces.

### Adapting for other physics domains
The Ising model theme can be swapped if students work in different areas:
- **Quantum computing:** replace Ising simulation with a simple variational quantum eigensolver (VQE); the Theorist derives the Hamiltonian, the Experimentalist runs Qiskit, the Analyst fits the ground state energy
- **Astrophysics:** replace with N-body simulation; the Theorist predicts orbital dynamics, the Experimentalist runs the simulation, the Analyst computes Lyapunov exponents
- **Biophysics:** replace with a protein folding energy landscape; the Theorist predicts folding pathways, the Experimentalist runs molecular dynamics, the Analyst identifies metastable states
The agent architecture and multi-agent coordination patterns remain identical — only the MCP tools and domain knowledge change.
