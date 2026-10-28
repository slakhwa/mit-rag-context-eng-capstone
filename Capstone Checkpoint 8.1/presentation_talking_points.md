# Capstone Design Review — Talking Points

Condensed from the full speaker-notes script on each slide into bullet points,
so you can glance and speak naturally instead of reading verbatim. Each slide
section below maps 1:1 to a slide in `Capstone_Design_Review_Presentation_completed.pptx`.

---

## Slide 1 — Intro

- Hi, I'm Sumeet — capstone design review for the **Research Paper Navigator**, a retrieval-augmented LLM system.
- Built incrementally across 8 checkpoints this program.
- Roadmap: the problem it solves → how it works → how I evaluated it → key engineering decisions & trade-offs → what's still left to improve.

## Slide 2 — The Problem

- **Hook:** LLMs answer with total confidence — even when they're completely wrong.
- Not hypothetical — proved it in **Checkpoint 1.1** on this exact corpus (153 CS papers: program synthesis, PL, neuro-symbolic AI).
- A bare LLM, no retrieval, hallucinated three different ways:
  - Invented a wrong title for a real paper
  - Fabricated a verbatim quote that doesn't exist anywhere in the text
  - Made up a precise-sounding performance statistic
- Every wrong answer was stated with full, unhedged confidence.
- **Stakes:** for someone who needs citable answers, a confidently wrong answer is more dangerous than no answer at all — that's the problem this system removes.

## Slide 3 — How It Works

- Five-stage pipeline, built up incrementally.
- **1) Sanitize** input → **2) Route** (simple question → direct 1-call path; complex → agent loop) → **3) Retrieve**: BM25 keyword index over all 153 papers, ~15,500 chunks (250 words each, 50-word overlap).
- Complex questions enter the **agent loop**: a planner checks what's been retrieved, decides if it's enough or issues a new targeted search — up to 3 rounds.
- **4) Answer** generation is strictly grounded in the retrieved document block; the user's own question text is treated as plain data, never as instructions.
- That data-vs-instruction boundary is the backbone of the security hardening (comes back in a later slide).

## Slide 4 — Evaluation & Evidence

- **Checkpoint 2.1** baseline: plain BM25 — fast, free, deterministic; strong fit for a fixed academic corpus with precise technical vocabulary.
- **Checkpoint 3.1** built the evaluation harness I reused for the rest of the program: an LLM-judge scoring pass/fail against grading notes I wrote myself, on the **same fixed 5-question set** every time — this is what makes the before/after comparisons honest, not cherry-picked.
- **Checkpoint 4.1 key result** — diagnosed a real retrieval failure: a question about a specific 2018 Ellis et al. paper never retrieved that paper's own text, because a different, more keyword-dense paper that merely cited it ranked higher.
  - Fix: a citation + same-author graph built from the corpus's own filename conventions and in-text citation mentions, disambiguated by year.
  - The fix worked exactly as designed on the targeted question — but on the same eval set, a different previously-passing question then **failed**, because swapping in the related paper evicted a chunk that question depended on.
  - Net result: **4 out of 5, both before and after.** Framed honestly — not a failed fix, but proof that retrieval improvements under a fixed context budget are trade-offs, not free wins.
- Flag forward: a citation-year misattribution bug first showed up here and was still present as late as 7.1 (covered in limitations).

## Slide 5 — Key Engineering Decisions

- **BM25 over embeddings** — this corpus has precise, stable technical vocabulary (author/algorithm names) where keyword matching is strong; semantic retrieval was a deliberate next-iteration item, not a hidden gap.
- **Citation graph, not a new ML model** — lightweight and explainable, built directly from the corpus's own filename conventions and citation mentions. Hand-verified (e.g., confirmed "Ellis et al. 2018" connects only to the real 2018 paper, not every paper by that author).
- **Same document budget for every technique** — every new retrieval method was constrained to use the same budget as the baseline, to keep comparisons fair. This is exactly what exposed the Checkpoint 4.1 trade-off.
- **No single complete fix (7.1 lesson)** — extending the input sanitizer closed two real gaps, but on the same test case it also increased token cost by 65% and revived a citation bug the unsanitized version had actually avoided. Reported honestly instead of claiming a clean win.

## Slide 6 — Limitations & Future Work

- **Citation-year misattribution (unresolved)** — first appeared in 5.1, confirmed again as late as 7.1: system can answer about a specific 2018 paper using content from a different paper by the same authors. This is a retrieval-and-reasoning correctness problem, not a security or cost problem — none of the Module 6/7 work targets it.
- **BM25's sensitivity to query verbosity** — the exact same information need can rank a target document anywhere from **8th to 28th** purely based on phrasing (confirmed by direct scoring). Matters because the 7.1 sanitizer neutralizes an attack's trigger words but can leave enough other words behind to still dilute the ranking — a blocklist fixes instruction-following, not verbosity.
- **Agent non-determinism** — found cases where the identical question, run twice, took a different number of search steps and reached a different conclusion.
- **If continuing, next 3 priorities:**
  1. Port the citation graph's year-disambiguation directly into the main retrieval path.
  2. Add a core-query-extraction step before BM25 — the only approach found so far that actually addresses verbosity dilution.
  3. Replace the exact-match cache with a semantic one, so a rephrased repeat question still gets the cost benefit.

## Slide 7 — Closing

- Thank you — happy to take questions.
- Glad to go deeper on the design rationale, the evaluation evidence, or the engineering trade-offs.
- Full implementation, worksheets, and all referenced evidence are in the GitHub repo linked on the slide.
