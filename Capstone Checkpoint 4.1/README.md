# Capstone Checkpoint 4.1 — Advanced Retrieval Implementation

**Status:** ✅ Complete — real citation/author graph built over the 153-paper corpus, targeted failure fixed, full before/after comparison run, worksheet filled in.

**Scenario (same as 1.1/2.1/3.1): Research Paper Navigator.**

## Approach

**Graph-based retrieval** over a citation/same-author graph (not query
decomposition — see rationale below). Nodes are papers; two edge types:

- **`same_author`** — two papers share a first author, extracted from each
  paper's filename (DBLP's own bibkey convention encodes it, e.g.
  `EllisMSST18` → "Ellis"). 274 edges.
- **`cites`** — paper A's full text literally mentions another corpus
  paper's first author as a citation (e.g. "Ellis et al."), **year-
  disambiguated**: a 4-digit year near the mention must match the specific
  candidate paper's year before the edge is added, so "Ellis et al. 2018"
  (in-corpus) doesn't get confused with "Ellis et al. 2021" (not in-corpus).
  106 edges.

`graph_retrieve()` takes the baseline's top-k BM25 seeds, then swaps the
single weakest seed for the best graph-connected related paper not already
present — capping total documents at the same `k` as the baseline, so the
two are directly, fairly comparable. `answer_from_docs()` labels every
document in the prompt as `[PRIMARY (keyword match)]` or `[RELATED CONTEXT
(via same_author/cites: <name>)]`, per the checkpoint's explicit instruction.

## The targeted failure (from Checkpoint 3.1)

Q4 ("Compare Ellis et al.'s EC2 and Bowers et al.'s Stitch, naming both
algorithms") — BM25 never retrieved Ellis et al.'s own 2018 paper; its best
chunk ranked 122nd of ~15,500 for this query, because Bowers et al.'s paper
discusses the topic far more densely (under the different name "DreamCoder").
I confirmed with the LLM-judge that **decomposition does not fix this** (ran
the demo: every sub-query still retrieves Bowers over Ellis) — the failure is
that one document is categorically unreachable by keyword match, which only
a relationship-aware method fixes.

## Results — before vs. after on the Checkpoint 3.1 eval set

| Query | Baseline | Graph |
|---|---|---|
| Q1 (counterfeit conundrum) | PASS | PASS |
| Q2 (RNA splicing) | PASS | PASS |
| Q3 (compression metric) | **PASS** | **FAIL** |
| Q4 (EC2 vs. Stitch — targeted) | **FAIL** | **PASS** |
| Q5 (off-corpus abstention) | PASS | PASS |
| **Total** | **4/5** | **4/5** |

**The headline result is not "4/5 unchanged" — it's Q3/Q4 trading places.**
Graph retrieval fixed the exact failure it targeted (Q4), but introduced a
real regression (Q3): evicting Bowers' weakest-scoring chunk to make room
for a related Ellis paper happened to remove the one chunk containing the
specific "Rewritten corpus size ratio (DreamCoder / Stitch)" detail Q3
needed — confirmed by reading that chunk directly. This is a genuine,
demonstrated cost of retrieving under a fixed document budget, not noise.

## Contents

- **`Required_Capstone_Checkpoint_4_1_Worksheet_completed.docx`** — the
  graded deliverable, including the honest Q3/Q4 trade-off analysis above.
- **`capstone_checkpoint_4_1_advanced_retrieval_starter.py`** — adapted to
  the real corpus: real BM25 baseline (same as 2.1/3.1), the citation/author
  graph builder, `graph_retrieve()`, `my_advanced_plan()` filled in, and a
  "Step 5b" (added beyond the original TODOs) that reruns the full
  Checkpoint 3.1 eval set through both baseline and graph retrieval for a
  fair before/after.
- **`checkpoint_4_1_responses.log`** — full evidence: every query's
  retrieved documents (with source labels), judge verdicts, and answers.

**Not included** (regenerate locally, see Checkpoint 2.1's README):
`paper_text/` — same corpus used throughout, kept out of git.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install python-dotenv langchain-openai langchain-core rank-bm25 networkx
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
# paper_text/ is NOT included (gitignored); regenerate per Checkpoint 2.1's README.
python capstone_checkpoint_4_1_advanced_retrieval_starter.py
```
