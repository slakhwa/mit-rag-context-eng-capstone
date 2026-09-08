r"""Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation
# **MO-LLM Module 2 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# In Checkpoint 1.1, you showed that a plain LLM can't reliably answer questions about
# your corpus. Now, you will **add retrieval**: Design a retrieval strategy for your scenario
# and build a **baseline retrieval system** that finds the most relevant documents for
# a query, so the model can ground its answers in them.
#
# This mirrors the Module 2 labs — keyword (BM25), vector (semantic), and hybrid
# retrieval — applied to your own capstone corpus. The graded deliverable is the completed 
# Capstone Checkpoint 2.1 worksheet, which includes your written responses and evidence of your 
# retrieval system implementation and testing. This script provides a small working example of 
# baseline retrieval. Use it to understand the retrieval workflow, then adapt the code to implement 
# and test a baseline retriever using your selected capstone dataset.
#
# **Learning outcomes (Module 2):**
# 1. Design a retrieval strategy appropriate for a given dataset and query type.
# 2. Implement and test a baseline retrieval system using structured and/or semantic
#    approaches.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1.
#
# | Scenario | Corpus | Retrieval considerations |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | long documents; you'll likely chunk them; questions often name a specific paper or compare papers. |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | many short-to-medium articles; questions name a figure/place or span several articles. |
#
# A good baseline is keyword (BM25), semantic (embeddings + vector search), or a
# hybrid of both — exactly what you built in Labs 1.2–2.2.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program. This checkpoint uses
#  the `openai/gpt-5.4-mini` model, with usage covered by the course credits. (this uses the paid gpt-5.4-mini chat model — covered by your course credits — and a keyword retriever, no embeddings).
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). Your real baseline (over your full corpus) is what you describe in
# the writeup.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.2
TOP_K = 3
LOG_PATH = Path.cwd() / "checkpoint_2_1_retrieval.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this course, put it in a .env file next to this "
            "script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## My REAL corpus: 153 research papers, chunked and BM25-indexed
#
# I replaced the starter's 6-document toy example with a real baseline over my
# actual capstone corpus (`Labs/CapstoneDatasets/ResearchPapers/`, ~150 PDFs).
#
# Pipeline:
# 1. Extract text from every PDF with `pdftotext` (poppler), cached to
#    `paper_text/<paper>.txt` (not committed — regenerate locally; see README).
# 2. Split each paper into ~250-word chunks with 50-word overlap (a paper is far
#    too long to hand to the LLM whole, and chunking is what makes retrieval
#    return a *relevant passage* instead of a whole 10-page PDF).
# 3. Build one BM25Okapi index over all chunks from all papers (a real
#    implementation of the "keyword ranking function that scores documents by
#    term frequency and inverse document frequency" the course describes,
#    rather than the starter's naive shared-word-count stand-in).
#
# No embeddings are used here — per this checkpoint's setup notes, semantic /
# hybrid retrieval is Module 2's later labs, not this baseline.

# %%
TEXT_DIR = Path(__file__).parent / "paper_text"
CHUNK_WORDS = 250
CHUNK_OVERLAP = 50


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _chunk_words(words: list[str], size: int, overlap: int) -> list[str]:
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, len(words), step) if words[i : i + size]]


def build_corpus() -> list[dict]:
    """Chunk every extracted paper into {paper, chunk_id, text} records."""
    chunks: list[dict] = []
    for path in sorted(TEXT_DIR.glob("*.txt")):
        words = path.read_text(encoding="utf-8", errors="ignore").split()
        for i, text in enumerate(_chunk_words(words, CHUNK_WORDS, CHUNK_OVERLAP)):
            chunks.append({"paper": path.stem, "chunk_id": i, "text": text})
    return chunks


# %% [markdown]
# ## Step 2 — The baseline retriever
#
# `retrieve()` scores every chunk with BM25 and returns the top-k. `answer()`
# then asks the LLM using only the retrieved chunk text, tagged by source paper
# so the model can (and is instructed to) cite/quote them.

# %%
print("Building BM25 index over the real corpus (chunking 153 papers)...")
CORPUS = build_corpus()
BM25_INDEX = BM25Okapi([_tokens(c["text"]) for c in CORPUS])
print(f"Indexed {len(CORPUS)} chunks from {len(set(c['paper'] for c in CORPUS))} papers.\n")


def retrieve(query: str, k: int = TOP_K) -> list[tuple[dict, float]]:
    """Baseline BM25 keyword retrieval over the real, chunked paper corpus."""
    scores = BM25_INDEX.get_scores(_tokens(query))
    ranked = sorted(zip(CORPUS, scores), key=lambda x: x[1], reverse=True)
    return [(chunk, score) for chunk, score in ranked[:k] if score > 0]


def answer(llm: ChatOpenAI, query: str, hits: list[tuple[dict, float]]) -> str:
    context = "\n\n".join(
        f"[{c['paper']} chunk {c['chunk_id']}] {c['text']}" for c, _ in hits
    )
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


# %% [markdown]
# ## Step 3 — Your representative queries (TODO)
#
# Submission item #2 asks for **3-5 representative queries** for your scenario and the
# results your system retrieves for each. Write those queries here. Some good ones include questions that:
#
# - Are answerable from **one** document (tests precision),
# - Need **several** documents (tests recall / aggregation),
# - Have wording that **differs** from the document's wording (i.e., tests whether
#   keyword vs. semantic retrieval matters for your corpus)
#
# Return a list of 3-5 query strings.

# %%
def my_representative_queries() -> list[str]:
    """5 representative queries for the Research Paper Navigator scenario,
    chosen to exercise different retrieval behaviors:
      1. Precise, single-doc, exact terminology (should nail one paper).
      2. Paraphrased away from the paper's own wording (tests keyword fragility).
      3. Precise, single-doc, different domain (out-of-cluster corpus check).
      4. A very recent (2025) paper, to contrast with Checkpoint 1.1's LLM
         hedging on this exact paper when it had no retrieval.
      5. Broad/conceptual, intended to span multiple papers (tests aggregation).
    """
    return [
        "What is the counterfeit conundrum in code language models?",
        "How does the paper measure whether library learning compression improves on DreamCoder?",
        "neural model of RNA splicing",
        "agent programming search over program execution paths",
        "comparing different approaches to library learning across papers",
    ]


# %% [markdown]
# ## Step 4 — Run the baseline and capture the evidence
#
# This runs each query through the baseline retriever and the LLM, printing the
# retrieved document ids/scores and the grounded answer, and logging everything to
# `checkpoint_2_1_retrieval.log`. The retrieved documents from the output are the rest of the evidence for
# submission item #2.

# %%
def run() -> None:
    llm = make_llm()
    queries = my_representative_queries()
    print(f"Checkpoint 2.1 — baseline retrieval  |  scenario: {SCENARIO}\n")
    for i, query in enumerate(queries, 1):
        hits = retrieve(query, TOP_K)
        hits_desc = [
            f"{c['paper']}#chunk{c['chunk_id']} (score={score:.2f})" for c, score in hits
        ]
        print("=" * 72)
        print(f"QUERY {i}: {query}")
        print(f"  retrieved: {hits_desc}")
        if not hits:
            print("  (nothing matched — note this in your writeup)")
            log(f"QUERY {i}: {query}", "retrieved=[] (no match)")
            continue
        ans = answer(llm, query, hits)
        print(f"  answer: {ans}\n")
        snippets = "\n".join(f"  - {c['paper']}#chunk{c['chunk_id']}: {c['text'][:200]}..." for c, _ in hits)
        log(f"QUERY {i}: {query}", f"retrieved={hits_desc}\nsnippets:\n{snippets}\nanswer={ans}")
    print("=" * 72)
    print("Done. Use the retrieved document results above as evidence in your writeup, and "
          "describe your REAL baseline (over your full corpus) in the submission.")


run()

# %% [markdown]
# ## Step 5 — Your written submission (the graded deliverable)
#
# Use your completed retrieval implementation and test results to complete the Capstone Checkpoint 2.1
# worksheet. In the worksheet, you will document your retrieval approach, provide evidence that your 
# system is functioning, include 3–5 representative queries and retrieved results, and reflect on where
# your approach performs well and where it struggles.  
 
# Save your completed Python file in the appropriate checkpoint folder in your GitHub repository. 
# Upload the completed worksheet only to the learning platform as your graded submission.
