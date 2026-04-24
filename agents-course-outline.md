# From Single Agents to Multi-Agent Systems

## A Hands-On Course for Physics, Mathematics & Computer Science Researchers

**Duration:** ~6–7 hours total — **SA Part** (Single-Agent): ~4 hours, **MAS Part** (Multi-Agent Systems): ~2–2.5 hours.
**Audience:** PhD students and Postdocs in physics, mathematics, neuroscience, and computer science. Background in ML/DL assumed.
**Stack:** Python 3.11+, Ollama (local LLMs), smolagents (HuggingFace), CrewAI, MCP protocol, a hand-rolled A2A message bus.
**Physics scenario:** Studying the phase transition of the 2D Ising model — from literature review to simulation to multi-agent cross-validation.

**Philosophy:**
1. Every concept is introduced incrementally — each module adds one capability on top of the previous one.
2. Students always see the code, never just the abstraction.
3. A single coherent physics use case threads through both parts: by the end, students have built a multi-agent physics research team that derives predictions, runs simulations, cross-checks results, and runs over a typed message-bus with a machine-verifiable isolation contract.

---

## Pre-Course Setup

Students must have installed before the SA Part:

- **Python 3.11+** with a virtual environment
- **Ollama** with a model pulled. The course was built against **`qwen2.5:7b`** (recommended if ≥16 GB RAM). Smaller models (`qwen2.5:3b`, `llama3.2:3b`) work for the SA Part but strain on the multi-agent modules.
- **Python packages:** see `setup/requirements.txt`
- **Course GitHub repo** cloned (contains starter notebooks, MCP server code, paper corpus)

> **Deliverables to build:**
> - `setup/requirements.txt` — all Python dependencies
> - `setup/setup_env.sh` — creates venv, installs packages, pulls the Ollama model
> - `setup/download_papers.py` — fetches 5–10 arXiv papers on the 2D Ising model (stores PDFs + metadata in `data/papers/`)
> - `setup/verify_setup.py` — checks Ollama is running, the model responds, packages are importable

---

## The Physics Scenario

The **2D Ising model** is the running example for the entire course. It was chosen because:

1. **It's familiar** to physics/math students (most have seen it in statistical mechanics)
2. **It has exact solutions** (Onsager's T_c = 2J / ln(1+√2) ≈ 2.269 J/k_B, exact magnetisation |m|(T) = (1 − sinh(2/T)^{−4})^{1/8} for T<T_c) so agents' outputs can be verified quantitatively
3. **It is the shared benchmark** that lets us compare a single agent, a sequential team, a debate, and an A2A bus on the *exact same three-temperature task* — changing only the architecture
4. **The research workflow is complete:** literature → theory → simulation → cross-validation → reconciled report

The three temperatures the course pins on are **T ∈ {2.00, 2.20, 2.50}** — two below T_c (where Onsager's formula returns a real number) and one above (where it returns a complex number and the exact answer is simply |m| = 0). The MAS Part reuses this exact triplet so outcomes across Modules 2.1, 2.2, and 2.3 can be compared row-for-row.

---

### SA Part — The Single Agent: "Physics Research Assistant"

The SA Part agent is a **personal research assistant for a PhD student starting a project on the Ising model**. It grows progressively more capable through each module. Here is exactly what it can do after each step, with concrete example interactions:

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
> - Student: *"Run an Ising simulation on a 32×32 lattice at T = 2.0, T = 2.20, and T = 2.50 with 10,000 Monte Carlo steps each. Describe what happens to the magnetization."*
> - Agent calls `run_ising_simulation` three times, observes a high |m| below T_c, fluctuating |m| near T_c, and near-zero |m| above T_c.
> - The agent is now an **experimentalist** that can design and run simulations.

#### After Module 1.4 (Reflection) — The agent self-checks and learns from mistakes

The agent now has a "Critic" that reviews its answers for errors, and a "Reflexion memory" that stores lessons learned across tasks.

> **Example interaction (showing error correction):**
> - Student: *"Estimate the critical temperature of the 2D Ising model by running a simulation on a 16×16 lattice."*
> - Agent (first attempt): finds the specific-heat peak at T≈2.35 on the small lattice and initially reports that.
> - Critic: flags the 3.6% discrepancy with Onsager's exact value and attributes it to finite-size effects on a 16×16 lattice.
> - Agent (revised): recommends running on a larger lattice or applying finite-size scaling corrections.
> - Reflexion memory stores the lesson: *"small lattice sizes produce shifted T_c estimates. Always use L≥32 or apply finite-size scaling corrections."*

#### Module 1.5 — The complete Physics Research Assistant

All capabilities are assembled into one agent. It can handle complex, multi-step research tasks:

> **End-to-end demo task:**
> *"Search our paper corpus for recommended simulation parameters, then run simulations at T ∈ {2.00, 2.20, 2.50} on a 32×32 lattice. Report the magnetization at each temperature, and check whether your results are consistent with Onsager's predictions."*
>
> The agent: (1) searches papers via RAG for recommended parameters, (2) extracts values, (3) runs 3 simulations via MCP, (4) compares results with the exact T_c using the calculator, (5) self-checks via reflection.

**The single agent's architecture at the end of the SA Part:**

```
┌──────────────────────────────────────────────────────────────┐
│              Physics Research Assistant (SA Part)            │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐    │
│   │          ReAct Loop (smolagents)                    │    │
│   │   Thought → Action → Observation → ... → Answer     │    │
│   └──────────────────────┬──────────────────────────────┘    │
│                          │                                   │
│          ┌───────────────┼───────────────────┐               │
│          │               │                   │               │
│   ┌──────▼──────┐ ┌──────▼──────┐  ┌────────▼────────┐       │
│   │ Local Tools │ │  RAG Tools  │  │   MCP Tools     │       │
│   │             │ │             │  │                 │       │
│   │ • calculator│ │ • search_   │  │ • search_arxiv  │       │
│   │ • unit_     │ │   papers    │  │ • run_ising_    │       │
│   │   converter │ │ (ChromaDB)  │  │   simulation    │       │
│   └─────────────┘ └─────────────┘  └─────────────────┘       │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐   │
│   │ Reflection (within-task) + Reflexion (across-task)   │   │
│   │ Critic prompt reviews answers; lessons stored in JSON│   │
│   └──────────────────────────────────────────────────────┘   │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐   │
│   │ LLM Backend: Ollama (qwen2.5:7b)                     │   │
│   └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

**Why one agent is not enough (motivation for the MAS Part):**

The single agent works well for one question at a time, but the SA Part climax (Module 1.5 §6) surfaces four failure modes that Reflection cannot dig out of:

1. **Tool confusion.** Five tools in one context means ReAct has to pick the right one at every step; misfires compound.
2. **Context bloat.** Each Observation occupies the same window as the running plan. Long runs push the plan out.
3. **Role conflation.** One brain is simultaneously theorist, simulator, and librarian. It does all three badly.
4. **Critic-as-same-brain.** When the author and the reviewer share the same weights, blind spots stay blind.

These are the exact four threads the MAS Part picks up.

---

### MAS Part — The Multi-Agent System: "Physics Research Team"

In the MAS Part, we split the single agent into specialized agents that collaborate — pinned to **the same three-temperature Onsager task** the SA Part closed on, so the architectural change is the only variable. Each module is a different *shape* of multi-agent collaboration on that same task.

#### The built modules

| Module | Pattern | Agents | Tools | What it tests |
|---|---|---|---|---|
| 2.0 | CrewAI hello-world | 1 | 1 | The plumbing: CrewAI ↔ Ollama on `qwen2.5:7b` |
| 2.1 | Sequential pipeline | 3 (Theorist / Experimentalist / Scholar) | 1 each | Does role split beat the lone generalist? |
| 2.2 | Debate | 3 (Theorist / Skeptic / Judge) | 1 (calculator) | Does an adversarial second pass catch the Theorist's hallucination? |
| 2.3 | A2A with typed messages | Same 3 as 2.2 | Same | Does protocol-level isolation replace prompt-level pretence? |

Everything from the SA Part is reused: the MCP simulation server, the ChromaDB paper corpus, the `calculator` tool, and the `expected_output` discipline from Module 1.5. The MAS Part adds the **shape change** — sequential, then adversarial, then bus-mediated.

#### The Three Specialists (Module 2.1)

**Agent 1: The Theorist**
- **Role:** Senior Theoretical Physicist
- **Tool:** `calculator_tool` (ported from the SA Part)
- **Task:** produce a three-row Markdown table `| T | |m|_theory |` for T ∈ {2.00, 2.20, 2.50} using Onsager's formula.
- **Failure mode seen live:** on 7B, the Theorist *does* call the calculator but then hallucinates the final table — ignoring its own tool outputs. This is the controlled failure that Modules 2.2 and 2.3 then try to catch.

**Agent 2: The Experimentalist**
- **Role:** Senior Computational Physicist
- **Tool:** `run_ising_simulation` (MCP, ported from the SA Part)
- **Task:** run three Monte-Carlo simulations at the Theorist's three temperatures and extend the table with a `|m|_sim` column.

**Agent 3: The Scholar**
- **Role:** Research Scholar / Literature reviewer
- **Tool:** `search_papers` (ChromaDB, ported from the SA Part)
- **Task:** for each row, cite the paper passage from the local corpus that motivates or corroborates the comparison.

**Crew process:** `Process.sequential`, Theorist → Experimentalist → Scholar, one LLM shared across all three, `allow_delegation=False`, per-agent `max_iter` tuned to the expected tool-call count.

#### The MAS Part workflow

```
SA Part (Module 1.5)           MAS Part — three shapes on the same task
───────────────────            ──────────────────────────────────────────────

one agent, 5 tools             Module 2.1 — Sequential
Critic same weights                 Theorist → Experimentalist → Scholar
                                    (Process.sequential, tools split 1:1)

                               Module 2.2 — Debate
                                    Theorist → Skeptic → Judge
                                    (same weights, adversarial second pass,
                                     prompt discipline forces independence)

                               Module 2.3 — A2A
                                    Theorist ─[ClaimToReview]─▶ Skeptic
                                    Skeptic   ─[ComparisonTable]─▶ Judge
                                    Judge     ─[Verdict]────────▶ user
                                    (typed MessageBus, machine-verifiable
                                     isolation contract)
```

#### Why 3 agents > 1 agent for this task — what the live runs showed

| Aspect | Single Agent (SA Part) | Sequential Team (2.1) | Debate (2.2) | A2A (2.3) |
|---|---|---|---|---|
| **Tool invocations** | 1 MCP call on 1 T instead of 3 | Experimentalist calls 3× correctly | Skeptic re-derives with same calculator | Same as 2.2, with typed envelopes |
| **Role clarity** | One brain, three hats | Each agent, one job, one tool | Author vs critic separation | Machine-verifiable role/message contract |
| **Error detection** | Critic = same weights → misses | Errors propagate unchecked in pipeline | Adversarial brain catches the Theorist's hallucination (NO/NO/YES) | Same verdict as 2.2 via isolation |
| **What still strained** | 4 failure modes from §6 | No cross-check step | 7B Skeptic sycophancy until SCRATCH-lines discipline added | Small prompt-adherence breaks; verdict prose miscounts |

#### SA Part → MAS Part Component Mapping — nothing is thrown away

| SA Part building block | Reused in the MAS Part as | Where |
|---|---|---|
| `calculator_tool` | Theorist's sole tool, Skeptic's sole tool | 2.1, 2.2, 2.3 |
| `run_ising_simulation` (MCP) | Experimentalist's sole tool | 2.1 |
| `search_papers` (ChromaDB/RAG) | Scholar's sole tool | 2.1 |
| `expected_output` / `final_answer` contract | Per-agent task contract in CrewAI | 2.0–2.3 |
| Reflection / Critic pattern (Module 1.4) | Reified as a separate **Skeptic** agent (not a second call on the same weights) | 2.2, 2.3 |
| smolagents `@tool` decorator | One-line swap to `crewai.tools.tool` | 2.1 §3 |
| Ollama + `qwen2.5:7b` | Same LLM, configured once, shared across agents | 2.0–2.3 |

#### Concrete example — the live three-agent Debate (Module 2.2)

**Theorist output (hallucinated as expected):**
> | T | \|m\|_theory |
> |---|---|
> | 2.00 | 0.3175 |   *← wrong; tool returned 0.9113*
> | 2.20 | 0.1984 |   *← wrong; tool returned 0.7848*
> | 2.50 | 0.0000 |   *← right (above T_c)*

**Skeptic output (after SCRATCH-lines-before-table discipline):**
> `SCRATCH: mA = 0.9113`, `SCRATCH: mB = 0.7848`, `SCRATCH: mC = 0` (re-derived via `calculator_tool`, ignored the Theorist's table).
> | T | \|m\|_theorist | \|m\|_skeptic | agree? |
> |---|---|---|---|
> | 2.00 | 0.3175 | 0.9113 | NO |
> | 2.20 | 0.1984 | 0.7848 | NO |
> | 2.50 | 0.0000 | 0.0000 | YES |

**Judge output:**
> `VERDICT: RECONCILE.` 2-of-3 rows disagree; recommend trusting the Skeptic's re-derived column.

**Same task in Module 2.3 (A2A) — same verdict, now with an isolation contract:**
> `[52036c38]   theorist -> skeptic     kind=ClaimToReview    (116c)`
> `[cea4b114]    skeptic -> judge       kind=ComparisonTable  (321c)`
> `[3bd3c0cb]      judge -> user        kind=Verdict          (57c)`
> `OK: all three isolation invariants hold.`
> `VERDICT: RECONCILE. Two rows agree and one row disagrees.` *(same label as 2.2; same off-by-one in the prose count — a reproducible 7B-prose failure, not a one-off)*

---

## SA Part: Building a Single AI Agent (~4 hours)

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
  - Connect to Ollama via `LiteLLMModel(model_id="ollama_chat/qwen2.5:7b")`
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
- **Build an MCP server** (Python, using the `mcp` SDK) that exposes 2 physics-relevant tools:
  1. **`search_arxiv(query: str, max_results: int) → list[dict]`**
  2. **`run_ising_simulation(lattice_size: int, temperature: float, num_steps: int, algorithm: str) → dict`**
- **Connect MCP to smolagents** via `ToolCollection.from_mcp()`; the agent dynamically discovers the tools from the server at startup.
- **Discussion:** MCP vs hardcoded tools — when to use which. The composability advantage: anyone can publish an MCP server, any agent can consume it.

**Hands-on exercise:**
- Start the pre-built MCP server (`mcp_server/physics_tools_server.py`)
- Connect it to the smolagents agent and verify tool discovery (`tools/list`)
- Task: run simulations at T ∈ {2.00, 2.20, 2.50} on a 32×32 lattice; observe ordered / near-critical / disordered phases in the returned magnetisations.

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
  - After the agent produces a `Final Answer`, add a second LLM call with a "Critic" system prompt that checks for mathematical correctness, consistency with simulations, proper citations, and signs of hallucination.
  - If the critic identifies issues, the agent loops back and revises (max 2 retries). Implemented as a wrapper around the smolagents agent.
- **Reflexion (across-task learning, Shinn et al. 2023):**
  - After completing a task, the agent stores a `lesson_learned` in a JSON file.
  - On future tasks, past lessons are injected into the system prompt.
  - Implemented as a simple `ReflexionMemory` class that reads/writes a JSON file.
- **Demo: catching a finite-size effect error** on a 16×16 vs 64×64 lattice.
- **Honest caveat named in the notebook:** the Critic shares weights with the author; when the model has a blind spot, the Critic inherits it. This is exactly the thread the **MAS Part's Skeptic agent** (Module 2.2) picks up — *different prompt, different role contract, same weights, enforced independence* as a structural fix for what Reflection patches locally.

**Hands-on exercise:**
- Add reflection to the agent — run a task where it initially makes an error
- Add Reflexion memory — run a sequence of 3 tasks and show improvement
- Examine the reflection memory file — what lessons did the agent learn?

**Notebook:** `05_reflection_agent.ipynb`

**Outputs:** A `ReflexionMemory` class; the agent now self-corrects and learns across tasks.

> **Key takeaway:** Reflection turns a one-shot generator into an iterative reasoner. Reflexion adds learning across tasks. Same-weights reviewers are the ceiling; Module 2.2 lifts it.

---

### Module 1.5 — The Complete Physics Research Assistant (25 min)

**Concept:** Assemble everything into a single coherent agent. Recap the architecture. Preview the MAS Part.

**Content:**
- Architecture recap (build the diagram live)
- **End-to-end demo task:**
  *"Search our paper corpus for recommended simulation parameters, then run simulations at T ∈ {2.00, 2.20, 2.50} on a 32×32 lattice. Report the magnetization at each temperature and check whether your results are consistent with Onsager's predictions."*
- The agent should: (1) search papers via RAG, (2) extract recommended parameters, (3) run 3 simulations via MCP, (4) compare with theory using the calculator, (5) self-check via reflection.
- **Motivation for the MAS Part:** This works on a good day but surfaces the four failure modes from §6 above — tool confusion, context bloat, role conflation, critic-as-same-brain. The MAS Part picks up each one with an architectural change, not a prompt patch.

**Hands-on exercise:**
- Students run the end-to-end task with their fully assembled agent
- Examine the full trace: every thought, action, observation, and reflection
- Identify failure modes: where does the single agent struggle?

**Notebook:** `06_full_agent.ipynb`

**Outputs:** The complete Physics Research Assistant — this is the building block for the MAS Part.

> **Key takeaway:** A modern AI agent is five things: an LLM, a reasoning loop, knowledge retrieval (RAG), external actions (MCP tools), and self-correction (reflection). Nothing more, nothing less.

---

## MAS Part: Building a Multi-Agent System (~2–2.5 hours)

### Narrative arc

Start from the single agent built in the SA Part. Discover *why* multiple agents beat one generalist. Build a 3-agent physics research team with CrewAI (sequential), turn it into a **Debate** (adversarial), then replace CrewAI's implicit context-passing with an explicit **A2A message bus** and a machine-verifiable isolation contract.

---

### Module 2.0 — Why Multi-Agent? Motivation & Architecture Patterns (20 min)

**Concept:** When a single agent is not enough, and how to decompose a problem into collaborating agents.

**Content:**
- **The failure of the single agent:** revisit the end of the SA Part. The four failure modes named in Module 1.5 §6 set the agenda for the MAS Part.
- **Multi-agent design patterns** (with diagrams):
  1. **Pipeline / Sequential** — Agent A → Agent B → Agent C (built in Module 2.1)
  2. **Debate / Adversarial** — author vs critic vs judge (built in Module 2.2)
  3. **Protocol-mediated / A2A** — typed messages over an isolated bus (built in Module 2.3)
  4. **Hierarchical / Orchestrator, Swarm / Peer-to-peer** — named, on the reading list, not the build list.
- **Mapping to physics:**
  - Pipeline ≈ data processing chain in experimental physics (trigger → reconstruction → analysis)
  - Debate ≈ theory vs experiment confrontation (crucial for catching systematic errors)
  - Protocol-mediated ≈ lab instrument buses with a defined schema — any node can be swapped, the contract stays the same.
- **Introduce CrewAI:**
  - Why CrewAI: simpler than AG2/AutoGen (which is in maintenance mode), role-based metaphor maps to our physics team, Ollama-compatible, actively maintained.
  - Core abstractions: **Agent** (role + goal + backstory + tools), **Task** (description + expected_output + agent), **Crew** (agents + tasks + process type).
  - Process types used in this course: `sequential` (2.1 and 2.2). For 2.3 we step out of CrewAI's process abstraction and run single-agent Crews driven by a hand-rolled message bus.

**Hands-on:** a ~30-line CrewAI hello-world with a single agent + a single task on `qwen2.5:7b` — just to prove the plumbing talks to your local Ollama box.

**Notebook:** `07_multi_agent_intro.ipynb`

> **Key takeaway:** Multi-agent design is about decomposing a problem into roles with different expertise, tools, and evaluation criteria — mirroring how real research teams work.

---

### Module 2.1 — A Team of Specialists (30 min)

**Concept:** Build a 3-agent team that runs the **exact same three-temperature Onsager task** as the SA Part's Module 1.5 climax — and watch role split cure the tool-confusion and role-conflation failure modes while leaving one unfixed.

**Content:**

**Three agents, one tool each** (per the built notebook):

| Agent | Tool | Scope |
|---|---|---|
| **Theorist** | `calculator_tool` | Onsager's formula only; no sim, no corpus |
| **Experimentalist** | `run_ising_simulation` (MCP) | Three MC runs at T ∈ {2.00, 2.20, 2.50}; no theory, no corpus |
| **Scholar** | `search_papers` (ChromaDB) | One citation per row; no theory, no sim |

Shared `LLM` (`qwen2.5:7b`) across all three, `allow_delegation=False`, per-agent `max_iter` tuned to the tool-call arithmetic.

**The three tasks** commit to a **fixed output shape** (this is where the SA Part's `final_answer` / `expected_output` discipline pays its second dividend, now consumed not just by the human but by the next agent as context):

- Theorist → a three-row Markdown table `| T | |m|_theory |`.
- Experimentalist → the same table extended with `|m|_sim`.
- Scholar → the same table extended with `citation`.

**Run the crew** and examine the output. **What the live run shows:**

- **Tool invocations right.** Experimentalist calls `run_ising_simulation` three times (once per T) — the lone generalist in 1.5 §6 collapsed to one MCP call on one temperature. Role split cured that.
- **Role conflation gone.** Each agent does its one job.
- **What still strains.** The Theorist *does* call `calculator_tool` but then hallucinates the final table anyway — a 7B "tool-output-ignored-at-final-answer" failure that a sequential pipeline with no cross-check step cannot catch. That unfixed mode is exactly the door into Module 2.2.

**Hands-on exercise:**
- Students implement the 3-agent crew from scratch using CrewAI, connecting to Ollama via CrewAI's `LLM` class.
- Port each SA Part tool by swapping `@smolagents.tool` → `@crewai.tools.tool` (a one-line change; the function body is untouched).
- Run the crew on the three-temperature task.
- **Exercise 2.1.1 — Harden the Experimentalist's task contract.** The SA Part taught us `qwen2.5:7b` will invent `T=` as a kwarg if the prompt does not forbid it. Add a "Practical Notes" block with explicit *Do NOT* rules.
- **Exercise 2.1.2 — Swap only the Theorist's brain** to `llama3.1:8b` and observe whether a different weight family hallucinates the same way.

**Notebook:** `08_multi_agent_team.ipynb`

**Outputs:** A working 3-agent CrewAI physics team that reuses every SA Part tool, and a concrete unfixed failure mode to hand off to Module 2.2.

> **Key takeaway:** Role split fixes tool confusion and role conflation for free; it does not fix author-hallucinating-at-final-answer. That needs a different agent, not a different tool.

---

### Module 2.2 — Debate: the critic pattern (20 min)

**Concept:** Add a second agent whose *only* job is to re-derive and dispute, then an adjudicator whose *only* job is to rule. Test whether an adversarial second pass with the same model and tool catches the Theorist's hallucination.

**Content:**

**Three agents on a shrunk tool surface:**

| Agent | Tool | Role |
|---|---|---|
| **Theorist** | `calculator_tool` | Produces the three-row Onsager table (same as 2.1, same failure mode) |
| **Skeptic** | `calculator_tool` | Independently re-derives; builds a comparison table with an `agree?` column |
| **Judge** | *none* | Reads only the Skeptic's comparison; issues a fixed-vocabulary verdict (ACCEPT_THEORIST / RECONCILE / ESCALATE) |

Tool surface is deliberately shrunk to `calculator_tool` only — the point is independent re-derivation with the *same* tool, not a richer one. The Skeptic's task carries two design commitments:
1. An explicit *"pretend the Theorist's table is not visible to you"* clause, forcing context-independence at the prompt level.
2. A **SCRATCH-lines-before-table** requirement (`SCRATCH: mA = ...`, `SCRATCH: mB = ...`, `SCRATCH: mC = ...`), so the model cannot skip computation and pretend it happened. This is a hard-won lesson from the first build pass.

**Live-run outcome (Build + one polish pass):**

- **v1 (pre-polish):** canonical **Skeptic sycophancy**. Zero `calculator_tool` calls, `|m|_skeptic` column copied from the Theorist's claim, `agree?` all YES, Judge: `ACCEPT_THEORIST` on the hallucinated numbers. 74.2 s wall — fast, wrong.
- **v2 (SCRATCH discipline added):** Skeptic calls `calculator_tool` twice (A.1 and A.2, with T=2.50 correctly short-circuited), emits the SCRATCH lines, produces `NO / NO / YES`. Judge issues `RECONCILE` with rationale prose that miscounts (says "2 agreed, 1 disagreed" — actually 1/2). The verdict label drives the downstream decision; the rationale sentence is prose. 74.8 s wall.

**Hands-on exercise:**
- Students build the three agents, run the Debate, watch v1 fail sycophantically, apply the SCRATCH discipline, and re-run.
- **Exercise 2.2.1 — Swap only the Skeptic's brain** to `llama3.1:8b`. Does a different weight family catch more / differently?
- **Exercise 2.2.2 — A second round of Debate.** Loop `Crew.kickoff` until Judge returns `ACCEPT_THEORIST` or 3 rounds hit.
- **Bridge to 2.3 (named in §6 of the notebook):** *prompt-engineered isolation vs protocol-enforced isolation.* The "pretend that table is not visible to you" clause is a patch. The protocol-level fix is to make it structurally true that the Skeptic never sees the Theorist's table. That is Module 2.3.

**Notebook:** `09_debate_agent.ipynb`

**Outputs:** A working three-agent Debate crew that catches the Module 2.1 failure mode; a named 7B failure mode (Skeptic sycophancy) that SCRATCH discipline plugs; and the hypothesis for Module 2.3.

> **Key takeaway:** Multi-agent reliability comes from cross-validation between agents with different roles — but on a small model, prompt discipline earns its keep. The structural fix is in 2.3.

---

### Module 2.3 — A2A: protocol-enforced isolation (20 min)

**Concept:** Replace CrewAI's implicit context-passing with a **typed message bus** that makes the isolation contract a line of Python. Same agents as 2.2, same model, same task — different protocol.

**Content:**

**The whole A2A surface of the module is ~20 lines of Python:**

```python
@dataclass
class Message:
    sender: str
    receiver: str
    kind: str          # "ClaimToReview" | "ComparisonTable" | "Verdict"
    payload: str
    parent_id: Optional[str] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

class MessageBus:
    def __init__(self):
        self._inboxes: dict[str, list[Message]] = defaultdict(list)
        self.log: list[Message] = []
    def send(self, msg): self._inboxes[msg.receiver].append(msg); self.log.append(msg)
    def inbox(self, name, kind=None): ...
```

Each agent's "turn" is a single-agent `Crew` (`Crew(agents=[agent], tasks=[task])`) driven by a local `a2a_run()` helper — the Crew sees only the current task description, nothing else from upstream, because no upstream was passed in via `Process.sequential`. The `MessageBus` is the *only* channel.

**Design commitments (hand-rolled, not library-wrapped):**
- **No `python-a2a`, no HTTP, no sockets** — the pedagogical surface is the 20 lines above, not a library. `python-a2a` gets namechecked in §8 as the migration target for "when you outgrow the laptop".
- **Approach (a):** same agents, same task, different protocol. If the verdict matches 2.2, we have shown the rearrangement is a drop-in.
- **Isolation must be MACHINE-VERIFIABLE, not aspirational.** The notebook's §6 ends with three asserts on bus state; if any fails, the notebook raises.

**Live-run outcome (88.7 s wall on `qwen2.5:7b`):**

```
[52036c38]   theorist -> skeptic     kind=ClaimToReview    (116c)
[cea4b114]    skeptic -> judge       kind=ComparisonTable  (321c)
[3bd3c0cb]      judge -> user        kind=Verdict          (57c)
OK: all three isolation invariants hold.
VERDICT: RECONCILE. Two rows agree and one row disagrees.
```

Three isolation invariants assert-checked: Skeptic got exactly one `ClaimToReview`; Judge got exactly one `ComparisonTable` with no `ClaimToReview` leak; verdict routed from Judge to user. All pass. The Judge reproduces 2.2's `RECONCILE` — a clean A/B that the protocol is a drop-in for the prompt discipline. The +14 s over 2.2 paid for the Skeptic's wasted-tool-call recovery on T=2.50 (hit the complex-number branch of Onsager's formula, retried, recovered).

**What the protocol buys:**
1. **An inspectable contract.** `assert bus.inbox("judge", kind="ClaimToReview") == []` is a line of Python. You cannot write the equivalent against `Process.sequential`'s context-passing.
2. **Shorter, less defensive prompts.** The 22 words of "pretend the table is not visible" paid rent for what the protocol now enforces for free.
3. **A substrate for cross-process / cross-agent deployment.** `a2a_run` + `MessageBus` maps one-to-one onto `python-a2a`'s `A2AClient` / `A2AServer`. When you outgrow the laptop, you change `a2a_run` and *nothing else*.

**What the protocol does not buy:** the Theorist still hallucinates (node property, not edge property); the Judge still miscounts in prose (reproducibly!); authentication / integrity are not implemented (see Exercise 2.3.1).

**Hands-on exercise:**
- Students build the `Message` dataclass and `MessageBus`, smoke-test the bus, build the three turn functions, and run §6.
- **Exercise 2.3.1 — Tamper with a payload in flight.** Mutate a message in the bus between send and receive. The Judge rules on the tampered thing. Typed envelopes are not signed envelopes.
- **Exercise 2.3.2 — Add an auditor role.** Add a fourth agent whose inbox subscribes to *all* messages and logs them. Zero changes to the three existing agents.
- **Exercise 2.3.3 — Migrate to `python-a2a`** in ~40 lines: the `a2a_run` helper is the only thing that changes.

**Notebook:** `10_a2a_protocol.ipynb`

**Outputs:** A hand-rolled in-process A2A runtime, a machine-verifiable isolation contract, and a one-function migration path to `python-a2a` for out-of-process deployment.

> **Key takeaway:** Protocol-level isolation replaces prompt-level pretence with a machine-verifiable contract. That closes the MAS Part and, with it, the course.

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

### SA Part Notebooks

| File | Module | Description |
|------|--------|-------------|
| `notebooks/01_naked_llm.ipynb` | 1.0 | Raw Ollama API calls, temperature experiments |
| `notebooks/02_react_agent.ipynb` | 1.1 | ReAct from scratch → smolagents, calculator + unit converter |
| `notebooks/03_rag_agent.ipynb` | 1.2 | Full RAG pipeline (PDF → chunks → embeddings → ChromaDB → tool) |
| `notebooks/04_mcp_tools.ipynb` | 1.3 | MCP server + client, arXiv search + Ising simulation |
| `notebooks/05_reflection_agent.ipynb` | 1.4 | Reflection critic + Reflexion memory, finite-size effect demo |
| `notebooks/06_full_agent.ipynb` | 1.5 | Complete Physics Research Assistant, end-to-end three-temperature demo |

### MAS Part Notebooks

| File | Module | Description |
|------|--------|-------------|
| `notebooks/07_multi_agent_intro.ipynb` | 2.0 | Architecture patterns + first CrewAI hello-world |
| `notebooks/08_multi_agent_team.ipynb` | 2.1 | 3-agent sequential team (Theorist / Experimentalist / Scholar) on the SA Part's three-temperature task |
| `notebooks/09_debate_agent.ipynb` | 2.2 | Debate pattern (Theorist / Skeptic / Judge), catches Theorist hallucination with SCRATCH discipline |
| `notebooks/10_a2a_protocol.ipynb` | 2.3 | Hand-rolled in-process A2A `MessageBus`; machine-verifiable isolation contract |

### Slides (Optional)

| File | Description |
|------|-------------|
| `slides/sa_part_slides.pdf` | Minimal lecture slides for the SA Part (mostly architecture diagrams) |
| `slides/mas_part_slides.pdf` | Minimal lecture slides for the MAS Part (MAS patterns, Debate, A2A diagrams) |

---

## Build Order

Recommended order for developing the materials (dependencies shown):

```
Phase 1 — Infrastructure (build first, everything depends on it):
  setup/requirements.txt
  setup/setup_env.sh
  setup/download_papers.py
  setup/verify_setup.py
  mcp_server/ising_simulator.py        ← the Monte Carlo code, tested standalone
  mcp_server/physics_tools_server.py   ← depends on ising_simulator.py

Phase 2 — SA Part Notebooks (build in order, each depends on previous):
  notebooks/01_naked_llm.ipynb
  notebooks/02_react_agent.ipynb       ← depends on 01 (chat function)
  notebooks/03_rag_agent.ipynb         ← depends on 02 (agent), setup/download_papers.py
  notebooks/04_mcp_tools.ipynb         ← depends on 03 (agent), mcp_server/
  notebooks/05_reflection_agent.ipynb  ← depends on 04 (full-tool agent)
  notebooks/06_full_agent.ipynb        ← depends on 05 (all components)

Phase 3 — MAS Part Notebooks (build in order):
  notebooks/07_multi_agent_intro.ipynb ← depends on SA Part concepts
  notebooks/08_multi_agent_team.ipynb  ← depends on 06 (reuses tools), CrewAI setup
  notebooks/09_debate_agent.ipynb      ← depends on 08 (CrewAI), same three-T task
  notebooks/10_a2a_protocol.ipynb      ← depends on 09 (same agents, different protocol)

Phase 4 — Slides (build last, after notebooks stabilize):
  slides/sa_part_slides.pdf
  slides/mas_part_slides.pdf
```

---

## References & Resources

### Key Papers
- Yao et al. (2023) — "ReAct: Synergizing Reasoning and Acting in Language Models"
- Shinn et al. (2023) — "Reflexion: Language Agents with Verbal Reinforcement Learning"
- Lewis et al. (2020) — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Du et al. (2023) — "Improving Factuality and Reasoning in Language Models through Multiagent Debate"
- Onsager (1944) — "Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition"

### Repositories
- [smolagents](https://github.com/huggingface/smolagents) — HuggingFace's minimal agent library (~1000 lines of core logic)
- [CrewAI](https://github.com/crewAIInc/crewAI) — Role-based multi-agent framework
- [rasbt/mini-coding-agent](https://github.com/rasbt/mini-coding-agent) — Minimal agent harness (reference architecture for Module 1.1)
- [A2A Protocol](https://github.com/a2aproject/A2A) — Agent-to-Agent open protocol (Linux Foundation) — migration target for Module 2.3 Exercise 2.3.3
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
- **Module 2.2 (Debate)** depends on students seeing the v1-sycophancy → v2-SCRATCH-discipline lesson land. Mitigation: the notebook ships both versions; if the live run happens to get lucky on v1, point at the committed v1 trace to show the failure mode.
- **Module 2.3 (A2A)** runs fully in-process — no networking, no FastAPI, no student-to-student coordination. No external risk.

### Model recommendations
- **`qwen2.5:7b`** — what the course was built against. Recommended if ≥16 GB RAM. Reliable tool-calling on the SA Part; surfaces the *interesting* failure modes on the MAS Part (Theorist hallucinating its own tool output, Skeptic sycophancy on v1, Judge miscounting in prose). These failures are pedagogically valuable and named explicitly in the §6/§7 of each notebook.
- `llama3.1:8b` — alternative weight family, used explicitly in Exercises 2.1.2 and 2.2.1 to test whether the failure modes are model-specific or universal.
- Smaller 3B–4B models work for the SA Part but strain on the MAS modules — they often skip tool calls entirely. Not recommended for the first-time run.
- The entire course is model-agnostic. Swapping models requires changing one string.

### The Ising model simulation
- The Monte Carlo code in `mcp_server/ising_simulator.py` should be simple, correct, and fast enough for classroom use.
- Metropolis algorithm: straightforward, ~50 lines of numpy. Good for small lattices.
- Wolff cluster algorithm: faster near T_c, ~80 lines. Recommended for larger lattices.
- Target: a 32×32 lattice with 10,000 MC steps should complete in <5 seconds on any laptop.
- The simulation returns pre-computed observables (M, E, C_v, χ) so the agent doesn't need to compute them.

### SA Part → MAS Part continuity
- The MCP server from the SA Part runs unchanged in the MAS Part — the CrewAI Experimentalist agent calls the same tools.
- The RAG pipeline from the SA Part runs unchanged — the CrewAI Scholar agent queries the same ChromaDB.
- The `calculator_tool` from Module 1.1 / 1.4 becomes the Theorist's and Skeptic's tool in Modules 2.1–2.3 via a one-line decorator swap.
- Reflection (Module 1.4) is conceptually what the Skeptic does in 2.2 — but as a *separate agent* with a separate prompt, rather than a second call on the same weights.
- Students should keep their SA Part virtual environment and MCP server running. The MAS Part builds on top, never replaces.

### Adapting for other physics domains
The Ising model theme can be swapped if students work in different areas:
- **Quantum computing:** replace Ising simulation with a simple variational quantum eigensolver (VQE); the Theorist derives the Hamiltonian, the Experimentalist runs Qiskit, the Scholar cites the VQE literature.
- **Astrophysics:** replace with N-body simulation; the Theorist predicts orbital dynamics, the Experimentalist runs the simulation, the Scholar cites chaos / Lyapunov-exponent results.
- **Biophysics:** replace with a protein folding energy landscape; the Theorist predicts folding pathways, the Experimentalist runs molecular dynamics, the Scholar cites the relevant folding-funnel literature.
The agent architecture and multi-agent coordination patterns remain identical — only the MCP tools and the domain corpus change.
