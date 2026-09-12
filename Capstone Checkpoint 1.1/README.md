# Capstone Checkpoint 1.1 — Evaluating When Retrieval Is Required

**Status:** ✅ Complete — worksheet filled in, baseline probes run, evidence logged.

**Capstone scenario (kept for the whole program): Research Paper Navigator** —
~150 computer-science research papers (program synthesis / PL / neuro-symbolic AI,
2005–2025), sourced from `Labs/CapstoneDatasets/ResearchPapers/` on the course
platform. Not included in this repo (large binary corpus, course-provided —
see `.gitignore`).

## Contents

- **`Required_Capstone_Checkpoint_1_1_Worksheet_completed.docx`** — the graded
  deliverable: the official worksheet, filled in with the scenario selection,
  two prompt/output examples, limitation analysis, pattern analysis, retrieval
  justification, retrieval requirements, and reflection — all grounded in the
  evidence from `checkpoint_1_1_responses.log`.
- **`capstone_checkpoint_1_1_baseline_starter.py`** — the checkpoint's baseline
  script (course-provided), completed with `my_probe_prompts()`: four original
  prompts targeting specific, verifiable facts in real papers from the corpus
  (a paper title, a verbatim abstract sentence, a cross-paper comparison, and a
  very recent/obscure paper), in addition to the three example prompts.
  Deliberately uses **no retrieval** — the point of this checkpoint is to
  establish the no-retrieval baseline.
- **`checkpoint_1_1_responses.log`** — the raw prompt/response evidence from
  running the baseline script. Several outputs were manually checked against
  the source PDFs and found to be hallucinated (invented a paper title,
  fabricated a verbatim quote, invented precise statistics) — see the
  worksheet's "Analysis of Model Limitations" section for details.
The related, earlier **Lab 1.1** exercise (chatbot with/without conversation
memory — not itself a graded checkpoint) now lives under
[`Labs/Module1/lab_1_1_chatbot_solution.py`](../Labs/Module1).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install python-dotenv langchain-openai langchain-core
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env

# Checkpoint 1.1 (no-retrieval baseline probes; requires the ResearchPapers
# corpus from the course platform to be downloaded locally to check answers
# against ground truth)
python capstone_checkpoint_1_1_baseline_starter.py
```
