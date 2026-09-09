r"""Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.

This demonstration system is not your capstone system and 
does not use the Research Paper Navigator or Wikipedia corpus.
"""

# %% [markdown]
# # Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis
# **MO-LLM Module 3 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# #
# This mirrors **Lab 3.1** (an LLM-judge that scores answers pass/fail against grading
# notes), applied to your capstone system. 
# You have a baseline retrieval system from Checkpoint 2.1. In this checkpoint, 
# you will use a structured evaluation approach to measure baseline performance, diagnose strengths 
# and weaknesses, and examine whether the evaluation framework detects problematic outputs.
# The starter script includes a small demonstration corpus and retriever to illustrate the 
# evaluation workflow. Apply the same evaluation approach to your selected capstone scenario and 
# baseline retrieval system. The graded deliverable is the completed Capstone Checkpoint 3.1 worksheet.
#
# **Learning outcomes (Module 3):**
# 1. Define evaluation metrics that reflect the real-world performance requirements of an LLM-powered retrieval system.
# 2. Identify key variables that influence system performance during evaluation and development.
# 3. Use language models to support evaluation tasks while avoiding common pitfalls.
# 4. Evaluate the performance of a retrieval-augmented system during development using a structured evaluation framework.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario and baseline retriever** from Checkpoints 1.1 and 2.1.
#
# | Scenario | Corpus |
# |---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) |

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12.**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# Runs on a tiny built-in sample corpus, so you do not need to prepare your own dataset.
# It still requires an OpenRouter API key to run the LLM (it is not offline or free of API
# calls). You apply the same evaluation approach to your real system for the report.

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
LOG_PATH = Path.cwd() / "checkpoint_3_1_evaluation.log"

SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)
JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES describing "
    "what a correct answer must contain. Reply with exactly one word: 'pass' if the "
    "answer satisfies the grading notes, or 'fail' if it does not."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this program, put it in a .env file next to this "
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
# ## My REAL Checkpoint 2.1 baseline system (not the starter's toy demo)
#
# The starter's built-in 6-document demo exists only to illustrate the judge
# mechanism. I'm applying the same evaluation approach to my actual capstone
# baseline instead: the BM25 keyword retriever over my full 153-paper corpus
# built in Checkpoint 2.1 (chunked ~250 words, 50-word overlap, indexed with
# rank_bm25). See Checkpoint 2.1's README for the extraction/chunking details.

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
    chunks: list[dict] = []
    for path in sorted(TEXT_DIR.glob("*.txt")):
        words = path.read_text(encoding="utf-8", errors="ignore").split()
        for i, text in enumerate(_chunk_words(words, CHUNK_WORDS, CHUNK_OVERLAP)):
            chunks.append({"paper": path.stem, "chunk_id": i, "text": text})
    return chunks


print("Building BM25 index over the real Checkpoint 2.1 corpus (153 papers)...")
CORPUS = build_corpus()
BM25_INDEX = BM25Okapi([_tokens(c["text"]) for c in CORPUS])
print(f"Indexed {len(CORPUS)} chunks from {len(set(c['paper'] for c in CORPUS))} papers.\n")


def retrieve(query: str, k: int = TOP_K) -> list[tuple[dict, float]]:
    scores = BM25_INDEX.get_scores(_tokens(query))
    ranked = sorted(zip(CORPUS, scores), key=lambda x: x[1], reverse=True)
    return [(chunk, score) for chunk, score in ranked[:k] if score > 0]


def answer(llm: ChatOpenAI, query: str, hits: list[tuple[dict, float]]) -> str:
    context = "\n\n".join(f"[{c['paper']} chunk {c['chunk_id']}] {c['text']}" for c, _ in hits)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


# %% [markdown]
# ## Step 2 — The evaluation metric (provided)
#
# A simple **LLM-judge** that returns pass/fail by checking an answer against grading
# notes — the same idea as Lab 3.1's DiscreteMetric, written directly here so the
# checkpoint needs no extra packages. Tuning this metric (stricter notes, a 'partial'
# level, a stronger judge model) is part of the diagnosis.

# %%
def judge(llm: ChatOpenAI, answer_text: str, grading_notes: str) -> str:
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}\n\nVerdict (pass/fail):"),
    ]
    verdict = llm.invoke(messages).content.strip().lower()
    return "pass" if "pass" in verdict else "fail"


# %% [markdown]
# ## Step 3 — Your evaluation set (TODO)
#
# Define the test set your evaluation runs on. Each item is a question plus
# **grading notes** — a short description of what a correct answer must contain (the
# judge checks the answer against these). Good evaluation sets include questions you
# expect to pass AND questions that probe known weaknesses.
#
# Return a list of 3-5 dicts: `{"question": "...", "grading_notes": "..."}`.

# %%
def my_eval_set() -> list[dict]:
    """5 evaluation items for the Research Paper Navigator baseline, chosen to
    reuse Checkpoint 2.1's evidence: 3 items where 2.1 showed retrieval works
    well (expected PASS), 1 item that revisits 2.1's known aggregation gap
    (expected FAIL), and 1 off-corpus abstention check (expected PASS)."""
    return [
        {
            "question": "What is the counterfeit conundrum in code language models?",
            "grading_notes": "Must state that counterfeit samples are incorrect programs that are still good enough to be generated at moderate temperature and pass weak/nontrivial correctness checks.",
        },
        {
            "question": "What does the neural model of RNA splicing use to improve interpretability?",
            "grading_notes": "Must mention the Motif and/or Aggregator components and that sparsity is used to improve interpretability.",
        },
        {
            "question": "How does the paper measure whether library learning compression improves on DreamCoder?",
            "grading_notes": "Must mention comparing rewritten corpus size / compression ratio between Stitch and DreamCoder (Eq. 18 / best-of-task metric).",
        },
        {
            # Expected FAIL: Checkpoint 2.1's Query 5 showed retrieval surfaces
            # Bowers et al. (Stitch) but not Ellis et al. 2018 (EC2) for broad
            # cross-paper "library learning" queries.
            "question": "Compare how Ellis et al.'s EC2 (NeurIPS 2018) and Bowers et al.'s Stitch approach library learning, naming both algorithms.",
            "grading_notes": "Must explicitly name BOTH 'EC2' (Ellis et al. 2018) AND 'Stitch' (Bowers et al. 2023) as the two algorithms being compared.",
        },
        {
            # Off-corpus abstention check.
            "question": "According to this corpus, who won the 2023 NBA Finals?",
            "grading_notes": "Must state that the documents do not contain this information, rather than guessing or fabricating an answer.",
        },
    ]


# %% [markdown]
# Run the demonstration code to understand the evaluation workflow. Then apply the same evaluation 
# design to your own capstone baseline system, using its retrieval and answer-generation functions. 
# Record results from your capstone system in the worksheet.
#
# ## Step 4 — Run the baseline evaluation
#
# Answers each question with the baseline retriever, scores it with the judge, and
# reports the pass rate. This is your baseline diagnosis: the failures are what you
# analyse in the report.

# %%
def run_evaluation() -> None:
    llm = make_llm()
    eval_set = my_eval_set()
    passes = 0
    print(f"Checkpoint 3.1 — baseline evaluation  |  scenario: {SCENARIO}\n")
    for i, item in enumerate(eval_set, 1):
        hits = retrieve(item["question"], TOP_K)
        hits_desc = [f"{c['paper']}#chunk{c['chunk_id']} ({s:.2f})" for c, s in hits]
        ans = answer(llm, item["question"], hits) if hits else "(no documents retrieved)"
        verdict = judge(llm, ans, item["grading_notes"])
        passes += verdict == "pass"
        print("=" * 72)
        print(f"Q{i}: {item['question']}")
        print(f"  retrieved={hits_desc}  verdict={verdict.upper()}")
        print(f"  answer: {ans}")
        log(f"Q{i}: {item['question']}", f"retrieved={hits_desc}\ngrading_notes={item['grading_notes']}\nverdict={verdict}\nanswer={ans}")
    print("=" * 72)
    print(f"Baseline pass rate: {passes}/{len(eval_set)}")


# %% [markdown]
# ## Step 5 — Validate the framework: can it catch a manipulated answer? (provided)
#
# A good evaluation framework must FAIL a wrong answer, not just pass good ones. This
# takes a question your corpus can answer, produces a correct answer, then feeds the
# judge a deliberately manipulated (false) answer — and checks that the judge flags it.
# This is your "evaluation framework validation" evidence for the report.

# %%
def validate_framework() -> None:
    llm = make_llm()
    # Real question from my corpus (Checkpoint 2.1 showed this retrieves cleanly).
    q = "What is the counterfeit conundrum in code language models?"
    notes = "Must state that counterfeit samples are incorrect programs that are still good enough to be generated at moderate temperature and pass weak/nontrivial correctness checks."
    good = answer(llm, q, retrieve(q))
    manipulated = (
        "The counterfeit conundrum refers to counterfeit money detection: the "
        "paper trains a computer-vision model to spot fake banknotes from scanned "
        "images, achieving 99.8% accuracy on the ECB Euro dataset."
    )
    good_verdict = judge(llm, good, notes)
    manip_verdict = judge(llm, manipulated, notes)
    print("\n--- Framework validation ---")
    print(f"  correct answer   -> {good_verdict.upper()}   (expected PASS)")
    print(f"  manipulated answer -> {manip_verdict.upper()}   (expected FAIL)")
    print("  The framework works if it PASSES the correct answer and FAILS the manipulated one.")
    log("FRAMEWORK VALIDATION", f"good={good_verdict} manipulated={manip_verdict}")


# %%
run_evaluation()
validate_framework()

# %% [markdown]
# ## Step 6 — Your written responses in the Capstone Checkpoint 3.1 worksheet
#
# Complete the Capstone Checkpoint 3.1 worksheet using evidence from your capstone evaluation.
# Address the seven sections: system overview, evaluation design, testing approach, baseline results, 
# performance analysis, evaluation framework validation, and reflection and next steps.
#
# 1. **System overview** — your scenario and your 2.1 baseline retriever.
# 2. **Evaluation design** — your criteria and metric (what "correct" means; how the
#    judge decides pass/fail; any thresholds).
# 3. **Testing approach** — how you built your evaluation set and ran it.
# 4. **Baseline results** — the pass/fail outcomes and where the system falls short.
# 5. **Performance analysis** — what the results reveal about strengths, weaknesses,
#    and failure modes.
# 6. **Evaluation framework validation** — show your framework detects a degraded or
#    manipulated output (use the Step 5 result, or your own).
# 7. **Reflection and next steps** — limitations of your evaluation and what you'll
#    improve (this motivates the advanced retrieval in Checkpoint 4.1).
