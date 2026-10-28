# MIT RAG & Context Engineering — Capstone Project

This repository tracks my ongoing capstone project for the MIT program's
retrieval-augmented generation (RAG) / LLM course. The capstone is built
incrementally: each module contributes a checkpoint that extends the same
underlying retrieval-augmented LLM system, from a bare chatbot in Module 1
toward a full RAG pipeline by the end of the program.

## The Final System

**Purpose.** A retrieval-augmented, agent-based question-answering system
("Research Paper Navigator") over 153 real computer-science research papers
(program synthesis, programming languages, neuro-symbolic AI). It answers
multistep, cross-document questions grounded in and quotable from the actual
paper text — built specifically to avoid the un-grounded hallucination
Checkpoint 1.1 demonstrated a bare LLM produces on this exact corpus.

**The canonical, most complete implementation** is
[`Capstone Checkpoint 7.1/capstone_checkpoint_7_1_production_ready_starter.py`](<./Capstone Checkpoint 7.1/capstone_checkpoint_7_1_production_ready_starter.py>)
— it is the cumulative, production-hardened version, incorporating every
prior checkpoint's retrieval, evaluation, agentic-loop, security, and
cost-optimization work. See that checkpoint's own
[README](<./Capstone Checkpoint 7.1/README.md>) for the full before/after
evidence behind each design decision.

**Install, configure, and run:**

```bash
cd "Capstone Checkpoint 7.1"
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install python-dotenv langchain-openai langchain-core rank-bm25

# Required environment variable — an OpenRouter API key:
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env

# Required local data — the corpus this system indexes. It is NOT committed
# to this repo (large, course-provided — see .gitignore). Place the 153
# paper PDFs' extracted text in a `paper_text/` subfolder here, one
# `<paper>.txt` per paper; see Capstone Checkpoint 2.1's README for the
# exact pdftotext extraction command used to build it originally.

python capstone_checkpoint_7_1_production_ready_starter.py
```

**Dependencies:** Python 3.11+, `python-dotenv`, `langchain-openai`,
`langchain-core`, `rank-bm25`. No GPU or vector database required — retrieval
is pure BM25 over locally-extracted text.

**Known limitations** (see Checkpoint 7.1's README for full evidence):
- A citation-year misattribution bug — the system can conflate a specific,
  named paper with a different, later paper by the same authors — persists
  unresolved across Checkpoints 5.1–7.1.
- BM25 retrieval quality is sensitive to query verbosity: the same
  information need can rank a target document anywhere from ~8th to ~28th
  purely based on phrasing.
- The agent loop is non-deterministic: identical input can take a different
  number of search steps and reach a different conclusion across runs.
- The input sanitizer is a regex trigger-phrase list, not a semantic
  classifier — generalized once already (Checkpoint 7.1) after verified
  gaps, but a sufficiently novel attack could still slip through.
- The cache is exact-string-match only; a rephrased repeat of the same
  question will not hit it.

**Capstone Checkpoint 8.1** (the design review presentation covering the
system's problem statement, architecture, evaluation evidence, engineering
trade-offs, and limitations) is in
[`Capstone Checkpoint 8.1/`](<./Capstone Checkpoint 8.1>).

## Repository organization

Each module's capstone checkpoint gets its own top-level folder, named
`Capstone Checkpoint <module>.<checkpoint>`. Inside each folder:

- **Solution file** — the completed code for that checkpoint.
- **Worksheet** — the lab/checkpoint worksheet or handout, if one was issued.
- A short `README.md` describing what the checkpoint covers and its status.

```
Capstone Checkpoint 1.1/
    capstone_checkpoint_1_1_baseline_starter.py
    Required_Capstone_Checkpoint_1_1_Worksheet_completed.docx
    checkpoint_1_1_responses.log
    README.md
Capstone Checkpoint 2.1/
    capstone_checkpoint_2_1_baseline_retrieval_starter.py
    Required_Capstone_Checkpoint_2_1_Worksheet_completed.docx
    checkpoint_2_1_retrieval.log
    README.md
Capstone Checkpoint 3.1/
    capstone_checkpoint_3_1_evaluation_starter.py
    Required_Capstone_Checkpoint_3_1_Worksheet_completed.docx
    checkpoint_3_1_evaluation.log
    README.md
Capstone Checkpoint 4.1/
    capstone_checkpoint_4_1_advanced_retrieval_starter.py
    Required_Capstone_Checkpoint_4_1_Worksheet_completed.docx
    checkpoint_4_1_responses.log
    README.md
Capstone Checkpoint 5.1/
    capstone_checkpoint_5_1_agent_rag_starter.py
    Required_Capstone_Checkpoint_5_1_Worksheet_completed.docx
    checkpoint_5_1_agent.log
    README.md
Capstone Checkpoint 6.1/
    capstone_checkpoint_6_1_security_cost_starter.py
    Required_Capstone_Checkpoint_6_1_Worksheet_completed.docx
    checkpoint_6_1_agent.log
    README.md
Capstone Checkpoint 7.1/
    capstone_checkpoint_7_1_production_ready_starter.py
    Required_Capstone_Checkpoint_7_1_Worksheet_completed.docx
    checkpoint_7_1_agent.log
    README.md
Capstone Checkpoint 8.1/
    Capstone_Design_Review_Presentation_completed.pptx
    recording_script.md
    README.md
```

As each new module is completed, its checkpoint folder is filled in with the
solution file and worksheet, following this same pattern. Note: the large
course-provided datasets (research papers / Wikipedia corpus) are kept local
only and are **not** committed to this repo — see `.gitignore`.

## Capstone scenario

**Research Paper Navigator** — selected in Checkpoint 1.1 and used for every
checkpoint going forward. The system answers questions over ~150 CS research
papers (program synthesis / PL / neuro-symbolic AI), quoting sources directly
and handling both single-paper and cross-paper questions.

## Checkpoint status

| Checkpoint | Topic | Status |
|---|---|---|
| [1.1](./Capstone%20Checkpoint%201.1) | Evaluating when retrieval is required (no-retrieval baseline) | ✅ Complete |
| [2.1](./Capstone%20Checkpoint%202.1) | Retrieval strategy design and baseline (BM25 keyword retrieval) | ✅ Complete |
| [3.1](./Capstone%20Checkpoint%203.1) | Evaluation infrastructure and baseline diagnosis (LLM-judge) | ✅ Complete |
| [4.1](./Capstone%20Checkpoint%204.1) | Advanced retrieval: citation/author graph over BM25 | ✅ Complete |
| [5.1](./Capstone%20Checkpoint%205.1) | Agent-based RAG: retrieve-decide-answer loop vs. fixed pipeline | ✅ Complete |
| [6.1](./Capstone%20Checkpoint%206.1) | Security and performance audit: injection probes, token cost, model ladder | ✅ Complete |
| [7.1](./Capstone%20Checkpoint%207.1) | Production hardening: sanitizer, caching, routing (mixed results) | ✅ Complete |
| [8.1](./Capstone%20Checkpoint%208.1) | Final system + design review presentation | ✅ Complete ([recording in Releases](https://github.com/slakhwa/mit-rag-context-eng-capstone/releases/tag/presentation-v1)) |

## Setup (per checkpoint)

Each checkpoint is a standalone Python script. General setup:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install python-dotenv langchain-openai langchain-core
```

Add a `.env` file (not committed — see `.gitignore`) with your OpenRouter API key:

```
OPENROUTER_API_KEY=sk-or-your-key-here
```
