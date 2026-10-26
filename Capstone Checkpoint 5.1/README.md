# Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System

**Status:** ✅ Complete — real agentic retrieval loop built over the 153-paper corpus, 3-way comparison (fixed / initial agent / refined agent) run on 3 multistep tasks, worksheet filled in.

**Scenario (same as 1.1–4.1): Research Paper Navigator.**

## Approach

An agentic retrieve → decide → (retrieve again | answer) loop (capped at 3
steps) over the same Checkpoint 2.1 BM25 index used throughout this project.
Each step's `decide()` call is shown the question, every query run so far,
and every chunk collected so far, and returns `{done, new_queries, reasoning}`
as JSON. The fixed Checkpoint 2.1 pipeline (one retrieval, one answer, no
loop) is reused unchanged as the baseline for a fair comparison.

## The initial evaluation — 3 multistep tasks

1. Compare Ellis et al.'s EC2 (2018) and Bowers et al.'s Stitch, naming both
   algorithms (the known-hard Checkpoint 3.1/4.1 case — EC2's paper ranks
   ~122nd by BM25 for this query).
2. Compare ENCOMPASS's PAN to DreamCoder's library learning (two distinct
   papers).
3. Do Gupta et al.'s 2024 and 2025 papers (same author) relate — tests
   same-author connection via agentic search alone, no citation graph.

**LLM calls, fixed vs. agent:** Task 1: 1 vs. 2. Task 2: 1 vs. 2. Task 3: 1
vs. 2–3 (the only task that actually triggered multistep behavior — Task 2's
single query happened to already cover both papers).

## Two diagnosed failures, two targeted refinements

1. **False early stopping.** `decide()` declared `done=true` on Task 1 after
   reasoning about "Ellis et al.'s neurally guided enumerative search" — a
   true fact that is **not** in the retrieved text at that point. It was
   judging sufficiency from its own background knowledge, not the shown
   documents. **Fix:** `REFINED_DECIDE_SYSTEM` explicitly forbids this and
   requires checking every named entity in the question against the shown
   text before declaring done.
2. **Near-miss retrieval.** Task 3's follow-up query ("What is the 2025 ICLR
   paper by Gupta...") ranked the target paper 8th by BM25 — just outside
   `k=3`. **Fix:** widened the agent's per-step `k` to 6.

## Results — and why this is more interesting than "refinement worked"

- **Task 1:** the refinement worked exactly as designed *behaviorally* — the
  refined agent now correctly keeps searching for all 3 steps instead of
  stopping after 1. But it did **not** make the answer more correct: with
  more steps and wider `k`, the refined agent retrieved more same-author
  Bowers/Ellis-citation chunks and confidently stated "EC2 (Ellis et al.,
  NeurIPS 2018) uses a deductive, bottom-up approach" — I verified
  word-for-word against the source PDF that this sentence actually describes
  **"Ellis et al. 2020, 2021,"** a different paper. The refined agent is more
  grounded (every word is real retrieved text) but now confidently **wrong**
  about which paper it describes — a worse outcome by trustworthiness than
  the initial agent's honest "I don't have enough information about EC2."
- **Task 2:** a genuine win — the refined agent found Ellis et al.'s own
  primary DreamCoder paper directly, instead of relying on Bowers' secondhand
  description of it.
- **Task 3:** **non-deterministic.** In one run, the initial agent's
  concise, keyword-dense follow-up query found the 2025 paper and gave a
  correct answer no fixed-pipeline call could reach. In another run (and for
  the refined agent in the same run), a longer, instruction-style query
  ranked the target paper 18th–28th by BM25 — confirmed by directly scoring
  the exact generated queries — well outside `k=6`. **The refinement did not
  reliably fix this**; if anything, its added prompt complexity may have
  made follow-up queries *more* verbose and less BM25-friendly.

**Bottom line:** the single biggest driver of success across all three tasks
was whether the LLM's follow-up query happened to be short and keyword-dense
— not more steps, not a wider `k`, not better stopping logic. Query phrasing
is an uncontrolled variable that swings retrieval quality more than the
architectural changes actually implemented here.

## Contents

- **`Required_Capstone_Checkpoint_5_1_Worksheet_completed.docx`** — the
  graded deliverable, including the full honest analysis above.
- **`capstone_checkpoint_5_1_agent_rag_starter.py`** — adapted to the real
  corpus/baseline, with `my_agent_plan()` filled in, `run_initial_evaluation()`
  (3-task fixed-vs-agent comparison) and `run_refined_evaluation()`
  (3-way fixed-vs-initial-vs-refined comparison) added beyond the original
  starter's single-demo-question `run()`.
- **`checkpoint_5_1_agent.log`** — full evidence: every task's fixed/initial/
  refined answers and complete step-by-step agent traces (queries run,
  documents collected, done/reasoning at each step).

**Not included** (regenerate locally, see Checkpoint 2.1's README):
`paper_text/` — same corpus used throughout, kept out of git.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install python-dotenv langchain-openai langchain-core rank-bm25
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
# paper_text/ is NOT included (gitignored); regenerate per Checkpoint 2.1's README.
python capstone_checkpoint_5_1_agent_rag_starter.py
```
