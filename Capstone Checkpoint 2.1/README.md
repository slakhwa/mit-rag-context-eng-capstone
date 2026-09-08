# Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation

**Status:** ✅ Complete — real BM25 baseline built over the full corpus, 5 queries tested, worksheet filled in.

**Scenario (same as Checkpoint 1.1): Research Paper Navigator.**

## Approach

Keyword-based (BM25) retrieval — no embeddings — per this checkpoint's own
setup notes, which explicitly scope Checkpoint 2.1 to a keyword baseline and
defer semantic/hybrid retrieval to later Module 2 labs.

Pipeline: all 153 paper PDFs → `pdftotext` → chunked into ~250-word windows
with 50-word overlap (15,501 chunks total) → indexed with `rank_bm25`'s
`BM25Okapi`. This replaces the starter script's toy 6-document example.

## Contents

- **`Required_Capstone_Checkpoint_2_1_Worksheet_completed.docx`** — the
  graded deliverable: retrieval approach + trade-offs, implementation
  evidence, 4 representative queries with retrieved results and observations,
  a strengths/limitations/patterns analysis, and a forward-looking reflection
  (recommending hybrid retrieval for the next checkpoint).
- **`capstone_checkpoint_2_1_baseline_retrieval_starter.py`** — the checkpoint
  script, adapted to build and query the real BM25 index described above
  instead of the starter's toy corpus, with `my_representative_queries()`
  implemented (5 queries covering precise single-doc, paraphrased, and broad
  conceptual/multi-doc retrieval).
- **`checkpoint_2_1_retrieval.log`** — full evidence: every query's retrieved
  chunk ids/scores, chunk text snippets, and the LLM's grounded answer.

**Not included** (regenerate locally, see below): `paper_text/` — the
pdftotext-extracted corpus this script indexes. Kept out of git like
`ResearchPapers/` in Checkpoint 1.1 (large, course-provided, and easily
regenerated).

## Key findings

- **Precise, terminology-heavy queries worked best.** Queries naming a
  distinctive phrase, technical term, or concept that appears in the source
  text (e.g. "counterfeit conundrum", "RNA splicing") retrieved the exact
  right paper(s) with no noise.
- **Retrieval fixed a Checkpoint 1.1 failure mode.** The plain LLM (no
  retrieval) hedged when asked about the 2025 ENCOMPASS paper in Checkpoint
  1.1 ("I don't have the paper text..."). With retrieval, the same question
  is answered correctly and immediately, grounded in the retrieved chunk.
- **Two real limitations surfaced.** (1) Incidental keyword overlap can pull
  in an unrelated paper (a query about DreamCoder pulled in an unrelated
  paper via the shared word "dreams"). (2) Broad, conceptual queries meant to
  span multiple papers tend to return several chunks from the single most
  keyword-dense paper rather than diversifying across genuinely distinct
  sources — motivating hybrid (BM25 + semantic) retrieval next.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install python-dotenv langchain-openai langchain-core rank_bm25
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env

# Regenerate the corpus this script indexes (requires poppler's pdftotext,
# and your local copy of Labs/CapstoneDatasets/ResearchPapers/ from the course):
mkdir -p paper_text
for f in /path/to/ResearchPapers/*.pdf; do
  pdftotext "$f" "paper_text/$(basename "${f%.pdf}").txt"
done

python capstone_checkpoint_2_1_baseline_retrieval_starter.py
```
