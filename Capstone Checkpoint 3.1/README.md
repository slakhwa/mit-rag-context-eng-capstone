# Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis

**Status:** ✅ Complete — LLM-judge evaluator built, 5-item eval set run against the real Checkpoint 2.1 baseline, framework validated, worksheet filled in.

**Scenario (same as 1.1/2.1): Research Paper Navigator**, evaluating the real
BM25 baseline from Checkpoint 2.1 — not the starter script's toy demo corpus.

## Approach

An LLM-judge (`openai/gpt-5.4-mini`, same model as the generator) scores each
answer **pass/fail** against one-sentence grading notes describing what a
correct answer must contain — the same pattern as Lab 3.1's discrete metric.

5-item evaluation set (`my_eval_set()`): 3 items reusing Checkpoint 2.1's
known retrieval successes (expected PASS), 1 item revisiting 2.1's known
cross-paper aggregation gap (expected FAIL), and 1 off-corpus question testing
whether the system abstains instead of hallucinating.

## Contents

- **`Required_Capstone_Checkpoint_3_1_Worksheet_completed.docx`** — the
  graded deliverable: system overview, evaluation design (criteria, metrics,
  and their limitations), testing approach, baseline results with 3 worked
  examples, performance analysis, framework validation, and reflection/next
  steps — all grounded in the real run below.
- **`capstone_checkpoint_3_1_evaluation_starter.py`** — the checkpoint
  script, adapted to evaluate my real Checkpoint 2.1 BM25 baseline (153-paper
  corpus) instead of the starter's toy 6-document demo. Implements
  `my_eval_set()` and replaces `validate_framework()`'s toy question with a
  real one from the corpus.
- **`checkpoint_3_1_evaluation.log`** — full evidence: every question's
  retrieved chunks, grading notes, verdict, and answer, plus the framework
  validation result.

**Not included** (regenerate locally, see Checkpoint 2.1's README):
`paper_text/` — the same pdftotext-extracted corpus used in Checkpoint 2.1,
kept out of git like there.

## Key findings

- **Baseline pass rate: 3/5 (60%).**
- **Two real, distinct failure modes**, not judge error (confirmed by the
  framework validation below) — one on each side of the pipeline:
  1. **Generation failure despite correct retrieval** — retrieval correctly
     returned the RNA-splicing paper's abstract chunk, which verbatim
     contains the required sentence ("separate Motif and Aggregator
     components ... enforcing sparsity to improve interpretability"). I
     confirmed this by hand (checking the exact word position against chunk
     boundaries): the fact was in the model's context the whole time. The
     model's answer simply drew from the *other* two retrieved chunks
     instead and never surfaced it — a synthesis/selective-attention issue,
     not a retrieval gap.
  2. **Primary vs. secondary source ranking** — asked to name both Ellis et
     al.'s "EC2" (2018) and Bowers et al.'s "Stitch," retrieval never
     surfaced Ellis et al.'s own paper at all (I confirmed its best-scoring
     chunk ranks 122nd overall for this query); it only found Bowers et
     al.'s paper, which cites Ellis's *later* system by a different name
     ("DreamCoder"). BM25's pure keyword density has no notion that a
     paper's own primary source should outrank a secondary paper that merely
     discusses it extensively. This one **is** a genuine retrieval gap.
- **Framework validation passed**: the judge correctly PASSed a real,
  grounded answer and FAILed a deliberately fabricated one on the same
  question — but this only tested an *obvious* fabrication. A noted blind
  spot: it's untested against a *subtle* wrong answer (right paper, wrong
  quote/number — the exact kind of hallucination found in Checkpoint 1.1).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install python-dotenv langchain-openai langchain-core rank_bm25
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
# paper_text/ is NOT included in this folder (kept out of git — see above);
# regenerate it locally following Checkpoint 2.1's README before running.
python capstone_checkpoint_3_1_evaluation_starter.py
```
