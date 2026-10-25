r"""Capstone Checkpoint 4.1 — Advanced Retrieval Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 4.1 — Advanced Retrieval Implementation
# **MO-LLM Module 4/Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This checkpoint advances your capstone system from a *single-pass* retriever to an
# **advanced retriever**. You will use the baseline retrieval system you built in
# Checkpoint 2.1 and evaluated in Checkpoint 3.1 to identify where single-pass retrieval
# falls short and apply the Module 4 techniques to fix it:
#
# - **Lab 4.1 — multistep retrieval (query decomposition):** Rewrite a complex
#   question into focused sub-queries, retrieve for each, and merge the results.
# - **Lab 4.2 — graph-based retrieval:** Organize your corpus as a graph of
#   relationships (citations, shared topics, links, same author/entity) and pull in
#   *related* documents a keyword/vector search would miss.
#
# You then **measure the improvement** against your 3.1 baseline. The graded
# deliverable is a **500-750 word written report** (final section). This script is a
# runnable demonstration of both techniques on a tiny sample corpus so you can see
# them work before adapting them to your real system.
#
# **Learning outcomes (Module 4):**
# 1. Identify common retrieval failures — e.g., complex multistep questions and too
#    much loosely-related context degrading the answer.
# 2. Explain *why* those failures occur with single-pass retrieval.
# 3. Implement an advanced retrieval strategy (query decomposition and/or graph
#    traversal) that addresses them.
# 4. Measure the improvement against your baseline using your 3.1 evaluation set.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Natural relationships to exploit in a graph |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | citations, shared authors, shared topics/keywords, "published-before" |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | hyperlinks between articles, shared categories, mentioned entities |
#
# Multistep decomposition shines on cross-document questions ("Compare X and Y,"
# "how did an idea evolve"); graph retrieval shines when the answer needs documents
# that are *related* to a hit but don't themselves match the query terms.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv networkx`
# 3. Use the OpenRouter API key provided for this course. This checkpoint uses
#   the `openai/gpt-5.4-mini` model, with usage covered by course credits.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This script runs on a tiny built-in sample corpus, so you do **not** need your full
# dataset indexed to complete it. You *will* refer to your real 2.1 system and 3.1
# evaluation results when you write the report.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # Latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.2
LOG_PATH = Path.cwd() / "checkpoint_4_1_responses.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

DECOMPOSE_SYSTEM = (
    "You are a query decomposition assistant for a document retrieval system. "
    "Break the user's question into 2-4 focused sub-queries that together cover "
    "everything needed to answer it. Each sub-query should target a distinct aspect. "
    'Return ONLY a JSON array of strings, e.g., ["sub-query 1", "sub-query 2"].'
)

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using only the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)

# Reused verbatim from Checkpoint 3.1, so the before/after comparison in Step 5b
# below scores baseline and advanced retrieval with the exact same judge.
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
            "OPENROUTER_API_KEY not set. Grab a free key at "
            "https://openrouter.ai/keys, put it in a .env file next to this "
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


def log_response(label: str, prompt: str, response: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    entry = (
        f"[{ts}]  {label}  SCENARIO={SCENARIO}  MODEL={LLM_MODEL}\n"
        f"PROMPT:   {prompt}\n"
        f"RESPONSE: {response}\n"
        f"{'-' * 72}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(entry)


# %% [markdown]
# ## My REAL Checkpoint 2.1/3.1 baseline (not the starter's toy demo)
#
# Same BM25 keyword retriever over my real 153-paper corpus, chunked identically
# to 2.1/3.1 (~250 words, 50-word overlap) so the baseline here is the exact same
# system already evaluated in Checkpoint 3.1 — required for a fair before/after.

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


print("Building BM25 index over the real corpus (chunking 153 papers)...")
CORPUS = build_corpus()
BM25_INDEX = BM25Okapi([_tokens(c["text"]) for c in CORPUS])
FULL_TEXT = {p.stem: p.read_text(encoding="utf-8", errors="ignore") for p in TEXT_DIR.glob("*.txt")}
print(f"Indexed {len(CORPUS)} chunks from {len(FULL_TEXT)} papers.\n")


def baseline_retrieve(query: str, k: int = 3) -> list[dict]:
    """Single-pass retrieval: my exact Checkpoint 2.1/3.1 BM25 baseline. Returns
    the top-k chunks, each a dict with paper, chunk_id, text, score, source='seed'."""
    scores = BM25_INDEX.get_scores(_tokens(query))
    ranked = sorted(zip(CORPUS, scores), key=lambda x: x[1], reverse=True)
    return [
        {"paper": c["paper"], "chunk_id": c["chunk_id"], "text": c["text"], "score": float(s), "source": "seed"}
        for c, s in ranked[:k] if s > 0
    ]


# %% [markdown]
# ## Step 2 — Multistep retrieval (provided, adapted from Lab 4.1)
#
# `decompose_query` asks the LLM to split the question into sub-queries. For each
# sub-query, we retrieve the top documents and **sum** each document's score across
# the sub-queries. A document relevant to several parts of the question rises to the
# top. (In your real system, replace `baseline_retrieve` with your 2.1 retriever.)

# %%
def decompose_query(llm: ChatOpenAI, query: str) -> list[str]:
    messages = [SystemMessage(content=DECOMPOSE_SYSTEM), HumanMessage(content=query)]
    raw = llm.invoke(messages).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("["): raw.rfind("]") + 1] if "[" in raw else raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed) and parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return [query]


def multistep_retrieve(llm: ChatOpenAI, query: str, k: int = 3) -> list[dict]:
    sub_queries = decompose_query(llm, query)
    print(f"  decomposed into: {sub_queries}")
    chunk_map: dict[tuple[str, int], dict] = {}
    for sq in sub_queries:
        for c in baseline_retrieve(sq, 3 * k):
            key = (c["paper"], c["chunk_id"])
            if key in chunk_map:
                chunk_map[key]["score"] += c["score"]
            else:
                chunk_map[key] = dict(c)
    ranked = sorted(chunk_map.values(), key=lambda c: c["score"], reverse=True)
    return ranked[:k]


# %% [markdown]
# ## Step 3 — Graph-based retrieval over my REAL corpus
#
# Nodes are papers. Two edge types, both derived directly from the real data
# (no manual labeling):
#
# - **`same_author`** — two papers share a first author. Found by extracting the
#   first author's last name from each paper's filename (DBLP's own bibkey
#   convention encodes it, e.g. `EllisMSST18` -> "Ellis"; non-DBLP filenames like
#   `Gupta2024` encode it directly). Verified by hand against all 153 filenames.
# - **`cites`** — paper A's full text literally mentions another corpus paper's
#   first author as a citation (e.g. "Ellis et al. 2018"). I require a 4-digit
#   year within ~20 characters of the mention that matches a specific candidate
#   paper's year (extracted the same way as the author name) before adding the
#   edge — this is what lets the system tell "Ellis et al. 2018" (in-corpus,
#   EC2) apart from "Ellis et al. 2021" (DreamCoder, not in this corpus) rather
#   than connecting to every same-author paper indiscriminately. I verified this
#   by hand: Bowers' paper mentions "Ellis et al." 26 times, but only the 2018
#   instances produce an edge once this year check is applied.
#
# Limitation I'm disclosing rather than hiding: when NO year is detectable near a
# mention, I do NOT add a fallback "cites" edge (unlike same_author) — I'd rather
# under-connect than add a low-confidence/likely-wrong edge. This trades recall
# for precision; see the worksheet's reflection on remaining limitations.

# %%
def first_author_lastname(stem: str) -> str | None:
    key = stem.split("_")[-1]
    if re.fullmatch(r"\d{4}(-\d+)?", key) or key.startswith("abs-") or re.match(r"^\d", key):
        return None
    m = re.match(r"^((?:[A-Z][a-z']+-?)+)[A-Z]*\d", key)
    if m:
        return m.group(1).rstrip("-")
    m = re.match(r"^([A-Za-z]+?)(?=\d)", key)
    if m:
        return m.group(1).capitalize()
    return None


def paper_year(stem: str) -> int | None:
    m = re.search(r"(19|20)\d{2}", stem)
    if m:
        return int(m.group(0))
    key = stem.split("_")[-1]
    m2 = re.search(r"(\d{2})$", key)
    if m2:
        yy = int(m2.group(1))
        return 2000 + yy if yy <= 50 else 1900 + yy
    return None


def build_graph() -> nx.DiGraph:
    paper_ids = list(FULL_TEXT.keys())
    author_of = {pid: first_author_lastname(pid) for pid in paper_ids}
    year_of = {pid: paper_year(pid) for pid in paper_ids}
    by_author: dict[str, list[str]] = {}
    for pid, author in author_of.items():
        if author:
            by_author.setdefault(author, []).append(pid)

    G = nx.DiGraph()
    for pid in paper_ids:
        G.add_node(f"paper:{pid}", node_type="paper")

    # same_author edges (bidirectional: co-authorship is symmetric)
    for author, papers in by_author.items():
        for p1 in papers:
            for p2 in papers:
                if p1 != p2:
                    G.add_edge(f"paper:{p1}", f"paper:{p2}", edge_type="same_author", via=author)

    # cites edges (directed: A cites B), year-disambiguated
    distinct_authors = sorted(by_author)
    for a_id, text in FULL_TEXT.items():
        a_author = author_of[a_id]
        for cited_author in distinct_authors:
            if cited_author == a_author:
                continue
            matches = list(re.finditer(
                rf"\b{re.escape(cited_author)}\b\s+et\s*al\.?|\b{re.escape(cited_author)}\s+and\s+[A-Z]", text
            ))
            if not matches:
                continue
            found_years = set()
            for m in matches:
                window = text[m.end(): m.end() + 20]
                found_years.update(int(y) for y in re.findall(r"(?:19|20)\d{2}", window))
            for target in by_author[cited_author]:
                if target != a_id and year_of[target] in found_years:
                    G.add_edge(f"paper:{a_id}", f"paper:{target}", edge_type="cites", via=cited_author)
    return G


GRAPH = build_graph()
_n_same_author = sum(1 for _, _, d in GRAPH.edges(data=True) if d["edge_type"] == "same_author")
_n_cites = sum(1 for _, _, d in GRAPH.edges(data=True) if d["edge_type"] == "cites")
print(f"Graph built: {GRAPH.number_of_nodes()} paper nodes, "
      f"{_n_same_author} same_author edges, {_n_cites} cites edges.\n")


def graph_retrieve(query: str, k: int = 3, reserve_slots: int = 1) -> list[dict]:
    """Advanced retrieval: baseline seeds, but with `reserve_slots` of the weakest
    seed chunks swapped out for the most relevant graph-connected neighbor paper(s)
    not already present — capping total returned at k, same as the baseline, so the
    two are directly comparable (same document budget, different selection)."""
    seeds = baseline_retrieve(query, k + reserve_slots)
    # Dedup to one (best) chunk per paper, preserving score order.
    seed_by_paper: dict[str, dict] = {}
    for c in seeds:
        seed_by_paper.setdefault(c["paper"], c)
    ordered_seed_papers = list(seed_by_paper.keys())

    related: list[dict] = []
    seen = set(ordered_seed_papers)
    for sp in ordered_seed_papers:
        node = f"paper:{sp}"
        if not GRAPH.has_node(node):
            continue
        for _, target, ed in GRAPH.out_edges(node, data=True):
            target_paper = target.removeprefix("paper:")
            if target_paper in seen:
                continue
            seen.add(target_paper)
            related.append({
                "paper": target_paper, "chunk_id": 0,
                "text": next(c["text"] for c in CORPUS if c["paper"] == target_paper and c["chunk_id"] == 0),
                "score": 0.0, "source": ed["edge_type"], "via": ed.get("via", ""),
            })

    kept_seed_papers = ordered_seed_papers[: max(0, k - min(len(related), reserve_slots))]
    result = [seed_by_paper[p] for p in kept_seed_papers]
    remaining = k - len(result)
    result.extend(related[:remaining])
    return result


# %% [markdown]
# ## Step 4 — Your advanced-retrieval plan (TODO)
#
# Design how you will apply these techniques to **your** capstone corpus. Return a
# dictionary with the keys below. This is the plan you will implement in your real system
# and describe in the report. Keep it concrete and specific to your scenario.

# %%
def my_advanced_plan() -> dict[str, Any]:
    """Return YOUR advanced-retrieval plan for your chosen scenario.

    TODO — your turn. Return a dictionary with these keys:
      - "technique":   "decomposition", "graph", or "both"
      - "node_types":  list[str]  — for graph: the kinds of nodes in your corpus
                       (e.g., ["paper", "author", "topic"] or ["article", "category"])
      - "edge_types":  list[str]  — the relationships you will surface
                       (e.g., ["cites", "same_author", "shares_topic"])
      - "test_queries": list[str] — 2-3 questions from your 3.1 evaluation set that
                       single-pass retrieval handled poorly and that should improve.
      - "rationale":   str — one or two sentences on why these choices fit your data.

    Delete the raise NotImplementedError line once your code works.
    """
    return {
        "technique": "graph",
        "node_types": ["paper"],
        "edge_types": ["same_author", "cites"],
        "test_queries": [
            "Compare how Ellis et al.'s EC2 (NeurIPS 2018) and Bowers et al.'s "
            "Stitch approach library learning, naming both algorithms.",
        ],
        "rationale": (
            "Checkpoint 3.1 (Q4) showed BM25 never retrieves a paper's own primary "
            "source when a secondary paper discusses it more densely under a "
            "different name (Bowers et al. ranked Ellis et al.'s 2018 paper 122nd "
            "overall). A citation/same-author graph fixes exactly this: it doesn't "
            "need the target paper to share the query's keywords, only to be "
            "graph-connected to a paper that already matched. Decomposition was "
            "explored in the demo above but wasn't the chosen fix here, since Q4's "
            "failure isn't that the question needs multiple sub-queries — it's "
            "that one needed document was never reachable by keyword match at all."
        ),
    }


# %% [markdown]
# ## Step 5 — Run baseline vs. advanced on the Checkpoint 3.1 failure case
#
# `answer_from_docs` explicitly labels each document as [PRIMARY] (a direct BM25
# keyword match — "seed") or [RELATED CONTEXT] (pulled in via a graph edge, with
# the edge type and the specific citation/author that justified it), per the
# checkpoint's instruction to distinguish primary vs. context documents in the
# prompt — so the LLM (and anyone reading the log) knows which documents actually
# matched the query versus which were added because they're connected to one.

# %%
def answer_from_docs(llm: ChatOpenAI, query: str, docs: list[dict]) -> str:
    parts = []
    for d in docs:
        if d["source"] == "seed":
            label = "PRIMARY (keyword match)"
        else:
            label = f"RELATED CONTEXT (via {d['source']}: {d.get('via', '')})"
        parts.append(f"[{label} — {d['paper']} chunk {d['chunk_id']}]\n{d['text']}")
    context = "\n\n".join(parts)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


def describe(docs: list[dict]) -> list[str]:
    return [f"{d['paper']}#chunk{d['chunk_id']} ({d['source']})" for d in docs]


def run_demo() -> None:
    llm = make_llm()
    query = (
        "Compare how Ellis et al.'s EC2 (NeurIPS 2018) and Bowers et al.'s Stitch "
        "approach library learning, naming both algorithms."
    )

    print("=" * 72)
    print(f"Checkpoint 4.1 demo — scenario: {SCENARIO}\nQuery: {query}\n")

    base = baseline_retrieve(query, 3)
    print(f"BASELINE (single-pass) retrieved: {describe(base)}")

    multi = multistep_retrieve(llm, query, 3)
    print(f"MULTI-STEP retrieved:             {describe(multi)}")

    g = graph_retrieve(query, 3)
    print(f"GRAPH retrieved:                  {describe(g)}")

    base_answer = answer_from_docs(llm, query, base)
    print(f"\nBaseline answer:\n{base_answer}\n")
    log_response("BASELINE", query, base_answer)

    graph_answer = answer_from_docs(llm, query, g)
    print(f"Graph-augmented answer:\n{graph_answer}\n")
    log_response("GRAPH", query, graph_answer)

    plan = my_advanced_plan()
    print("My advanced-retrieval plan:")
    print(json.dumps(plan, indent=2))

    print("=" * 72)


run_demo()

# %% [markdown]
# ## Step 5b — Re-run the full Checkpoint 3.1 evaluation set, baseline vs. graph
#
# Not part of the original starter's TODOs, but required by the worksheet's
# Step 5 ("re-run your 3.1 evaluation on baseline vs advanced... report
# before/after"). Reuses the exact same 5-item eval set and LLM-judge from
# Checkpoint 3.1 so the comparison is apples-to-apples.

# %%
def judge(llm: ChatOpenAI, answer_text: str, grading_notes: str) -> str:
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}\n\nVerdict (pass/fail):"),
    ]
    verdict = llm.invoke(messages).content.strip().lower()
    return "pass" if "pass" in verdict else "fail"


def checkpoint_3_1_eval_set() -> list[dict]:
    """Identical to my_eval_set() in Checkpoint 3.1 — copied here (not imported)
    so this script has no cross-folder dependency."""
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
            "question": "Compare how Ellis et al.'s EC2 (NeurIPS 2018) and Bowers et al.'s Stitch approach library learning, naming both algorithms.",
            "grading_notes": "Must explicitly name BOTH 'EC2' (Ellis et al. 2018) AND 'Stitch' (Bowers et al. 2023) as the two algorithms being compared.",
        },
        {
            "question": "According to this corpus, who won the 2023 NBA Finals?",
            "grading_notes": "Must state that the documents do not contain this information, rather than guessing or fabricating an answer.",
        },
    ]


def run_comparative_evaluation() -> None:
    llm = make_llm()
    eval_set = checkpoint_3_1_eval_set()
    base_passes, graph_passes = 0, 0
    print("=" * 72)
    print("Step 5b — Comparative evaluation: baseline vs. graph-augmented (3.1 eval set)\n")
    for i, item in enumerate(eval_set, 1):
        base_docs = baseline_retrieve(item["question"], 3)
        graph_docs = graph_retrieve(item["question"], 3)

        base_answer = answer_from_docs(llm, item["question"], base_docs) if base_docs else "(no documents retrieved)"
        graph_answer = answer_from_docs(llm, item["question"], graph_docs) if graph_docs else "(no documents retrieved)"

        base_verdict = judge(llm, base_answer, item["grading_notes"])
        graph_verdict = judge(llm, graph_answer, item["grading_notes"])
        base_passes += base_verdict == "pass"
        graph_passes += graph_verdict == "pass"

        print(f"Q{i}: {item['question']}")
        print(f"  baseline: {describe(base_docs)} -> {base_verdict.upper()}")
        print(f"  graph:    {describe(graph_docs)} -> {graph_verdict.upper()}")
        log_response(
            f"COMPARATIVE Q{i}",
            item["question"],
            f"baseline_retrieved={describe(base_docs)}\nbaseline_verdict={base_verdict}\nbaseline_answer={base_answer}\n\n"
            f"graph_retrieved={describe(graph_docs)}\ngraph_verdict={graph_verdict}\ngraph_answer={graph_answer}",
        )
        print()
    print("=" * 72)
    print(f"Baseline pass rate: {base_passes}/{len(eval_set)}")
    print(f"Graph pass rate:    {graph_passes}/{len(eval_set)}")


run_comparative_evaluation()

# %% [markdown]
# ## Step 6
#
# The checkpoint deliverable is your completed Capstone Checkpoint 4.1 worksheet, not
# code. Using evidence from your real system, cover:
#
# 1. **System overview** State your scenario and your 2.1 baseline retriever.
# 2. **Failure diagnosis** Describe the meaningful retrieval failure or limitation identified through
#    your Checkpoint 3.1 evaluation and explain why it occurs.
# 3. **Data redesign** Describe the node/edge schema you defined to expose relationships in
#    your corpus (citations/authors/topics, or links/categories/entities), and why you chose it. 
# 4. **Advanced retrieval implemented** Provide a query decomposition and/or graph traversal,
#    with code or log evidence. Label primary vs. context documents in the prompt.
# 5. **Comparative evaluation** Re-run your 3.1 evaluation on baseline vs advanced,
#    **comparing a similar number of retrieved documents**, and report before/after.
# 6. **Analysis & reflection** Discuss where it helped, where it hurt (redundant/over-broad
#    context), the cost/latency trade-off, and remaining limitations.
#
