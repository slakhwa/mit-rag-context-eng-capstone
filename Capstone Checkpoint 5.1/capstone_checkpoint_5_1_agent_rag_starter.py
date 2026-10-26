r"""Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System
# **MO-LLM Module 5 -  Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This is the final build step of your capstone. You will turn the retrieval system you've
# developed into an **agent-based RAG system**. Instead of a fixed pipeline, an agent
# decides at each step whether it has enough information, what to retrieve next, which
# tool to use, or whether to answer. This activity allows you to apply the Module 5 labs (Lab 5.1's agentic
# retriever and Lab 5.2's tool-using agent) to your capstone scenario.
#
# The graded deliverable is your completed Capstone Checkpoint 5.1 worksheet. This script is a
# runnable demonstration of an agentic loop on a tiny sample corpus so you can see the
# decision-making before adapting it to your full system.
#
# **Learning outcomes (Module 5):**
# 1. Build an agent-based system that integrates retrieval and external tools to complete user tasks.
# 2. Use system prompts to guide agent behavior and decision-making.
# 3. Evaluate an agentic workflow against a fixed retrieval pipeline.
# 4. Design workflows that coordinate retrieval, reasoning, and tool use within a RAG system.
#
# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Useful agent tools/actions |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs | semantic search; "find papers by author/year"; "follow citations"; ask a clarifying question |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles | semantic search; "find by category"; "follow links"; ask a clarifying question |
#
# An agent shines when one query isn't enough — it can search, look at what it found,
# and decide to search again (or use a different tool) before answering.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this course. The labs use the paid
#    `openai/gpt-5.4-mini` model, which is covered by the course credits.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). The built-in corpus is deliberately minimal: It only demonstrates
# the agent control flow, not retrieval quality, and it is unrelated to your capstone:
# Howerver, your design should target your own capstone corpus. Your full agent is what you describe
# in the written submission.

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

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
TEMPERATURE = 0.2
MAX_STEPS = 3
LOG_PATH = Path.cwd() / "checkpoint_5_1_agent.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

DECIDE_SYSTEM = (
    "You are an agent retrieving from a small document collection. Given the question, "
    "the queries already run, and the documents found so far, decide what to do next. "
    'Respond with ONLY a JSON object: {"done": true|false, "new_queries": ["..."], '
    '"reasoning": "..."}. Set done=true when you have enough to answer; otherwise give '
    "1-2 new_queries targeting what is still missing (do not repeat past queries)."
)
ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided documents, "
    "quoting where you can. If they do not contain the answer, say so."
)

# Refinement (see worksheet Step 3): my initial run showed DECIDE_SYSTEM let the
# model judge "enough to answer" using its own background knowledge rather than
# the actual retrieved text, causing a false done=True on Task 1 (it reasoned
# about "Ellis et al.'s neurally guided enumerative search" — true, but nowhere
# in the retrieved chunks). This adds the same "ground only in retrieved text"
# discipline ANSWER_SYSTEM already has, applied to the stopping decision itself.
REFINED_DECIDE_SYSTEM = (
    "You are an agent retrieving from a document collection. Given the question, "
    "the queries already run, and the documents found so far, decide what to do next. "
    "Judge sufficiency ONLY from the literal text of the documents shown below — do "
    "NOT use your own background knowledge to decide something is 'already known'; "
    "if a fact needed to answer is not explicitly present in the shown document text, "
    "treat it as missing, even if you personally know it. Identify every distinct "
    "named entity, paper, or sub-topic the question asks about, and check each one "
    "individually against the shown text before setting done=true. "
    'Respond with ONLY a JSON object: {"done": true|false, "new_queries": ["..."], '
    '"reasoning": "..."}. Set done=true only when every named part of the question is '
    "explicitly covered by the shown text; otherwise give 1-2 new_queries targeting "
    "what is still missing (do not repeat past queries)."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Grab a free key at https://openrouter.ai/keys, "
            "put it in a .env file next to this script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE,
                      api_key=check_api_key(), base_url=OPENROUTER_BASE_URL)


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## My REAL Checkpoint 2.1 baseline (not the starter's toy 6-doc demo)
#
# Same BM25 keyword retriever over my real 153-paper corpus, chunked identically
# to 2.1/3.1/4.1 (~250 words, 50-word overlap). This serves TWO roles here: it's
# the tool the agent calls at each decision step below, AND (called once, with
# no decision loop) it IS the "fixed Checkpoint 2.1 pipeline" the worksheet asks
# me to compare the agent against.

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
CHUNK_BY_ID = {f"{c['paper']}#chunk{c['chunk_id']}": c["text"] for c in CORPUS}
print(f"Indexed {len(CORPUS)} chunks from {len(set(c['paper'] for c in CORPUS))} papers.\n")


def retrieve(query: str, k: int = 3) -> list[str]:
    """Real BM25 retrieval, returning chunk ids like 'paper#chunkN'."""
    scores = BM25_INDEX.get_scores(_tokens(query))
    ranked = sorted(zip(CORPUS, scores), key=lambda x: x[1], reverse=True)
    return [f"{c['paper']}#chunk{c['chunk_id']}" for c, s in ranked[:k] if s > 0]


def fixed_pipeline_answer(llm: ChatOpenAI, question: str, k: int = 3) -> tuple[str, int]:
    """The 'fixed Checkpoint 2.1 pipeline': ONE retrieval call, ONE LLM call, no
    decision loop. Returns (answer, llm_call_count) for the cost comparison."""
    doc_ids = retrieve(question, k)
    context = "\n\n".join(f"[{i}] {CHUNK_BY_ID[i]}" for i in doc_ids)
    answer = llm.invoke([SystemMessage(content=ANSWER_SYSTEM),
                         HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}")]).content
    return answer, 1


def decide(llm: ChatOpenAI, question: str, collected: dict[str, str], executed: list[str],
           system_prompt: str = DECIDE_SYSTEM) -> dict:
    docs = "\n".join(f"[{i}] {CHUNK_BY_ID[i]}" for i in collected) or "(none yet)"
    user = f"Question: {question}\n\nQueries run: {executed or '(none)'}\n\nDocuments so far:\n{docs}"
    raw = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user)]).content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        d = json.loads(raw)
        return {"done": bool(d.get("done", True)), "new_queries": d.get("new_queries", []) or [],
                "reasoning": d.get("reasoning", "")}
    except (json.JSONDecodeError, ValueError):
        return {"done": True, "new_queries": [], "reasoning": "parse-fail -> stop"}


def agentic_answer(llm: ChatOpenAI, question: str, max_steps: int = MAX_STEPS,
                    decide_system: str = DECIDE_SYSTEM, retrieve_k: int = 3) -> dict:
    """Provided (adapted to the real retriever): a minimal agentic loop — retrieve,
    decide whether to continue, repeat. Returns a dict with the answer, the total
    LLM call count (1 per decide() + 1 for the final answer), and a step-by-step
    trace, so the before/after comparison can report cost, not just answer text.
    `decide_system`/`retrieve_k` are overridable so the refined run (Step 3) can
    swap in REFINED_DECIDE_SYSTEM and a wider per-step k without duplicating the
    whole loop."""
    collected: dict[str, str] = {}
    executed: list[str] = []
    pending = [question]
    llm_calls = 0
    trace: list[dict] = []
    for step in range(max_steps):
        for q in pending:
            for doc_id in retrieve(q, k=retrieve_k):
                collected[doc_id] = CHUNK_BY_ID[doc_id]
            executed.append(q)
        d = decide(llm, question, collected, executed, system_prompt=decide_system)
        llm_calls += 1
        print(f"  step {step + 1}: have {sorted(collected)}  -> done={d['done']}  ({d['reasoning'][:60]})")
        trace.append({"step": step + 1, "queries_run": list(pending), "collected": sorted(collected), **d})
        if d["done"] or not d["new_queries"]:
            break
        pending = d["new_queries"]
    context = "\n\n".join(f"[{i}] {collected[i]}" for i in collected)
    answer = llm.invoke([SystemMessage(content=ANSWER_SYSTEM),
                        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}")]).content
    llm_calls += 1
    return {"answer": answer, "llm_calls": llm_calls, "trace": trace, "docs_used": sorted(collected)}


# %% [markdown]
# ## Step 2 — Your agent design (TODO)
#
# Design the agent you will build for **your** scenario. Return a dictionary with the keys
# below — this is the plan you implement in your real system and describe in the report.

# %%
def my_agent_plan() -> dict[str, Any]:
    """Return your agent design for your chosen scenario.

    TODO — your turn. Return a dictionary with these keys:
      - "tools": list[str]      — the actions/tools your agent can take (e.g.,
                                  ["semantic_search", "find_by_author", "follow_citation",
                                   "clarify", "answer"]).
      - "stop_condition": str   — how the agent decides it has enough to answer.
      - "system_prompt_idea": str — one or two sentences on how you'll instruct the agent
                                  to choose actions (this is what "guiding agent behavior
                                  with system prompts" means).
      - "test_tasks": list[str] — 2-3 questions for your scenario that need more than one
                                  retrieval step (so the agentic loop earns its keep).

    Example (illustrative — replace with your own scenario):
        return {
            "tools": ["semantic_search", "find_by_author", "clarify", "answer"],
            "stop_condition": "the retrieved passages cover every part of the question",
            "system_prompt_idea": "Search first; clarify only if the request is ambiguous; "
                                  "answer once the retrieved text supports a grounded reply.",
            "test_tasks": [
                "What changed between the v1 and v2 proposals, and who approved it?",
                "Summarize the budget decisions discussed across Q1 and Q2.",
            ],
        }

    Delete the raise NotImplementedError line once your code works.
    """
    return {
        # Honest to what's actually implemented below — the provided loop only
        # supports "search again" or "answer"; it has no real human-in-the-loop
        # clarify action (this pipeline runs non-interactively), so I'm not
        # claiming a tool that doesn't exist in the code.
        "tools": ["search_corpus", "answer"],
        "stop_condition": (
            "The decide() call's LLM judges the retrieved chunks sufficient to "
            "answer every part of the question (done=true), OR MAX_STEPS "
            "iterations are reached — whichever comes first."
        ),
        "system_prompt_idea": (
            "Treat the first retrieval as a draft: check it against EVERY clause "
            "of the question, not just the most obvious one, and issue a "
            "genuinely different follow-up query (not a restatement of the same "
            "search) targeting specifically what's still missing, rather than "
            "declaring done once any relevant-looking text appears."
        ),
        "test_tasks": [
            "Compare how Ellis et al.'s EC2 (NeurIPS 2018) and Bowers et al.'s "
            "Stitch approach library learning, naming both algorithms.",
            "What problem does the ENCOMPASS paper's 'probabilistic angelic "
            "nondeterminism (PAN)' address, and how does that compare to "
            "DreamCoder's approach to library learning?",
            "What interpretability techniques does Gupta et al.'s 2024 Genome "
            "Biology paper use for RNA splicing, and does the 2025 ICLR paper "
            "by the same author (Gupta) build on this approach?",
        ],
    }


# %% [markdown]
# ## Step 3 — Run the agentic loop and capture the evidence
#
# Runs the provided agentic loop on a multistep question (watch it retrieve, decide,
# and retrieve again), then prints your plan. Reproduce this in your real system for the
# report and compare it against your fixed Checkpoint 2.1 retriever on the same task.

# %%
def run_initial_evaluation() -> None:
    llm = make_llm()
    plan = my_agent_plan()
    print(f"Checkpoint 5.1 — agentic RAG vs. fixed pipeline  |  scenario: {SCENARIO}")
    print("My agent plan:")
    print(json.dumps(plan, indent=2))
    print()

    for i, question in enumerate(plan["test_tasks"], 1):
        print("=" * 72)
        print(f"TASK {i}: {question}\n")

        fixed_answer, fixed_calls = fixed_pipeline_answer(llm, question)
        print(f"FIXED PIPELINE ({fixed_calls} LLM call):\n{fixed_answer}\n")

        print("AGENT (step-by-step):")
        result = agentic_answer(llm, question, max_steps=MAX_STEPS)
        print(f"\nAGENT ANSWER ({result['llm_calls']} LLM calls, docs={result['docs_used']}):\n{result['answer']}\n")

        log(
            f"TASK {i} (initial agent, MAX_STEPS={MAX_STEPS})",
            f"Q: {question}\n\n"
            f"FIXED ({fixed_calls} calls): {fixed_answer}\n\n"
            f"AGENT ({result['llm_calls']} calls, docs={result['docs_used']}): {result['answer']}\n\n"
            f"AGENT TRACE: {json.dumps(result['trace'], indent=2)}",
        )
    print("=" * 72)


run_initial_evaluation()

# %% [markdown]
# ## Step 3b — Refined agent: re-run the same 3 tasks (not in original starter)
#
# Two targeted refinements, each motivated by a specific observed failure above:
# 1. `REFINED_DECIDE_SYSTEM` — Task 1's agent declared done=true after reasoning
#    about "Ellis et al.'s neurally guided enumerative search," a true fact that
#    is NOT in the retrieved text (confirmed by reading the trace) — the decider
#    was using its own background knowledge, not the retrieved documents, to
#    judge sufficiency. The refined prompt forbids this explicitly.
# 2. `retrieve_k=6` (vs. the baseline's k=3) — Task 3's agent already asked the
#    right follow-up query ("What is the 2025 ICLR paper by Gupta...") every
#    single step, but the target paper ranks 8th for that exact query (confirmed
#    directly against the BM25 index) — just outside k=3. A wider per-step k
#    gives the agent's already-correct queries room to actually succeed.

# %%
def run_refined_evaluation() -> None:
    llm = make_llm()
    plan = my_agent_plan()
    print("=" * 72)
    print("Step 3b — Refined agent vs. initial agent vs. fixed pipeline\n")

    for i, question in enumerate(plan["test_tasks"], 1):
        print("=" * 72)
        print(f"TASK {i}: {question}\n")

        fixed_answer, fixed_calls = fixed_pipeline_answer(llm, question)

        print("INITIAL AGENT:")
        initial = agentic_answer(llm, question, max_steps=MAX_STEPS)

        print("REFINED AGENT:")
        refined = agentic_answer(llm, question, max_steps=MAX_STEPS,
                                 decide_system=REFINED_DECIDE_SYSTEM, retrieve_k=6)

        print(f"\nFIXED    ({fixed_calls} calls): {fixed_answer[:200]}...")
        print(f"INITIAL  ({initial['llm_calls']} calls, docs={initial['docs_used']}): {initial['answer'][:200]}...")
        print(f"REFINED  ({refined['llm_calls']} calls, docs={refined['docs_used']}): {refined['answer'][:200]}...\n")

        log(
            f"TASK {i} (three-way comparison)",
            f"Q: {question}\n\n"
            f"FIXED ({fixed_calls} calls): {fixed_answer}\n\n"
            f"INITIAL AGENT ({initial['llm_calls']} calls, docs={initial['docs_used']}): {initial['answer']}\n"
            f"INITIAL TRACE: {json.dumps(initial['trace'], indent=2)}\n\n"
            f"REFINED AGENT ({refined['llm_calls']} calls, docs={refined['docs_used']}): {refined['answer']}\n"
            f"REFINED TRACE: {json.dumps(refined['trace'], indent=2)}",
        )
    print("=" * 72)


run_refined_evaluation()

# %% [markdown]
# ## Step 4 — Your written submission (the graded deliverable)
#
# The deliverable is a **Capstone Checkpoint 5.1 worksheet** (suggested length: 500-750 words). Cover the following:
#
# 1. **System overview**: State your scenario and the retrieval system you built (2.1–4.1).
# 2. **Agent design**: Describe the actions/tools your agent can take, and how a **system prompt**
#    guides it to choose among them (retrieve, use a tool, clarify, or answer).
# 3. **Workflow**: Explain how retrieval, reasoning, and tool use are coordinated across steps,
#    and how the agent decides it has enough information to answer.
# 4. **Evaluation**: Compare the agent-based system against your fixed Checkpoint 2.1/3.1
#    pipeline on a few representative tasks. Did multi-step decision-making improve the
#    answers? At what cost (extra model calls, latency)?
# 5. **Refinements**: Describe any refinements you made to the agent (for example,
#   retrieval strategy, prompting, tool selection, stopping criteria, or other
#   workflow improvements) and explain how they affected system performance.
# 6. **Reflection**: When an agentic approach adds value vs. when a simpler pipeline is
#    better, and the limitations/trade-offs of increasingly autonomous systems.
#
# Include evidence (sample tasks, the agent's step-by-step decisions from a log, before/
# after comparison).
