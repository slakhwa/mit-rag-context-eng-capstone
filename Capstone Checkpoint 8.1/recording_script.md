# Recording Script — Capstone Design Review

Exported from the deck's speaker notes before they were removed for submission.
Quoted text is what you say; bracketed lines are delivery cues.

## Slide 1 — Cover

[Hold this slide ~5 seconds while you start speaking.]

OPEN: "Hello — welcome to my capstone design review."

Then advance to the next slide to introduce yourself.

## Slide 2 — Introduction

OPEN: "Hi, I'm Sumeet. This is my capstone design review for the Research Paper Navigator — a retrieval-augmented LLM system built incrementally across all eight checkpoints of this program."

ROADMAP (say as one flowing sentence): "Over the next few minutes I'll cover:
  1. the problem it solves,
  2. how the system works,
  3. how I evaluated it,
  4. the key engineering decisions and trade-offs behind it,
  5. and what's still left to improve."

TRANSITION: "Let me start with the problem."

## Slide 3 — The Problem

HOOK: "Large language models answer with total confidence — even when they're completely wrong."

[Pause one beat — let the hook land.]

PROOF: "That's not hypothetical for this project. In Checkpoint 1.1, I tested a bare LLM — no retrieval at all — on the exact corpus this system covers: 153 computer-science research papers in program synthesis, programming languages, and neuro-symbolic AI. It hallucinated three different ways:
  1. it invented a wrong title for a real paper,
  2. it fabricated a verbatim quote that doesn't exist anywhere in the text,
  3. and it made up a precise-sounding performance statistic comparing two systems.
Every one of those wrong answers was stated with full, unhedged confidence."

STAKES: "For a researcher or student who needs citable, trustworthy answers, a confidently wrong answer is more dangerous than no answer at all. That is the problem this system is built to remove."

TRANSITION: "So here's how the system removes it."

## Slide 4 — The Solution

OPEN: "At a high level, the system is a five-stage pipeline, built up incrementally across the program."

[Walk the diagram left to right, pointing at each numbered card.]

STAGE 1 — User Question: "A natural-language research question comes in."

STAGE 2 — Sanitize: "An input sanitizer strips injected instructions and escapes untrusted text."

STAGE 3 — Route: "A router decides: simple one-hop question, or a multistep one?"

STAGE 4 — Retrieve + Decide: "Retrieval runs against a BM25 keyword index over all 153 papers — about 15,500 passages of 250 words each, with 50-word overlap — extended with a citation graph. For multistep questions, an agent loop takes over: a planner looks at what's been retrieved, decides whether it's enough, and if not issues a new, targeted search — up to three rounds."

STAGE 5 — Answer: "A final generation call produces the response, strictly grounded in the retrieved documents. Everything else in the prompt — including the user's own question text — is treated as data, never as instructions."

TRANSITION: "That data-versus-instructions boundary is the backbone of the security hardening — I'll come back to it. First: how do I know this works?"

## Slide 5 — Retrieval and Evaluation

OPEN: "So — how do I know this works? Let me show you the evidence."

BASELINE (2.1): "I started simple. The baseline is plain BM25 keyword search. It's fast. It's free. And it gives the same answer every time. On academic papers full of precise technical terms, keyword matching gets you a long way."

EVALUATION (3.1): "Before improving anything, I built the measuring stick. Five representative questions. Written grading notes for each one. An LLM judge scores every answer — pass or fail. And here's the important part: that question set never changed. Same five questions, every checkpoint. That's what keeps my before-and-after comparisons honest."

KEY RESULT (4.1): "Now — the most interesting result of the whole program. I found a real retrieval failure. I asked about a specific 2018 paper by Ellis and colleagues. The system never retrieved that paper. Why? Because a different paper that merely cited it used the same keywords more often. So it ranked higher. My fix: a citation graph, built from the corpus itself, that follows the citation links to the real paper."

THE TWIST: "Did it work? Yes — on that question. But watch what happened. A different question, one that used to pass... started failing. The new paper I pulled in pushed out a chunk the other question needed. Final score: four out of five. Before the fix — and after. Same number."

[Pause one beat.]

LESSON: "With a fixed context budget, a retrieval improvement is never a free win. It's a trade."

FLAG FORWARD: "One more thing. There's a bug where the system confuses a 2018 paper with a later paper by the same authors. It showed up in Checkpoint 5.1. It was still there in 7.1. I'll come back to that."

TRANSITION: "These results drove every design decision. Let me walk you through the big four."

## Slide 6 — Engineering Decisions

OPEN: "Four decisions shaped this system. For each one, I'll give you the choice — and the reason."

DECISION 1 — keywords over embeddings: "First: keyword search over embeddings. Why? Look at what's in this corpus. Author names. Algorithm names. Exact technical terms. That's exactly where keyword matching is strongest — and where embeddings can actually do worse. So semantic search isn't missing from this system. It's deferred. Deliberately."

DECISION 2 — a graph, not a model: "Second: for citation-aware retrieval, I built a simple graph instead of training a model. It comes straight from the corpus's own filenames and citation mentions. And because it's simple, I could check it by hand. 'Ellis et al., 2018' links to the 2018 paper — not to everything Ellis ever wrote. I verified that before I trusted it."

DECISION 3 — same budget for everything: "Third: every new technique got the exact same document budget as the baseline. No quiet advantages. That rule is what exposed the trade-off I just showed you. And that's the point — a comparison that hides the cost isn't a comparison."

DECISION 4 — layered defenses: "Fourth: security. There is no single, complete fix — and I tested that directly. I extended my input sanitizer. It closed two real gaps. But the very same change raised token cost by sixty-five percent. And it brought back a citation bug the old version had avoided. So I chose layered, partial defenses — and I reported the side effects, instead of claiming a clean win."

TRANSITION: "Which brings me to what's still open."

## Slide 7 — Limitations and Future Improvements

OPEN: "No system like this is ever finished. So let me be specific about what's still broken."

LIMITATION 1 — citation-year misattribution: "Number one. The most stubborn bug: citation-year misattribution. Ask about a specific 2018 paper, and the system can answer from a different paper — same authors, different year. It appeared in Checkpoint 5.1. It was still there in 7.1. And none of my security or cost work touches it. Because it isn't a security problem. It's a correctness problem. It needs its own fix."

LIMITATION 2 — phrasing sensitivity: "Number two: phrasing. The same question, worded differently, can rank the target paper anywhere from 8th... to 28th. I measured that directly. And it has a security angle. My sanitizer strips an attack's trigger words — but the leftover words can still bury the right document. A blocklist stops instructions. It doesn't stop verbosity."

LIMITATION 3 — non-determinism: "Number three: the agent isn't deterministic. Same question. Two runs. A different number of searches — and a different conclusion."

NEXT PRIORITIES: "So if I kept going — three priorities. One: put year-disambiguation into the main retrieval path. Two: extract a short core query before search runs — the only approach I've found that actually beats verbosity. Three: a semantic cache, so a rephrased question still gets the savings."

TRANSITION: "And with that — let me close."

## Slide 8 — Questions

CLOSE: "That's the Research Paper Navigator. Thank you."

OFFER: "I'm happy to go deeper on any decision, any number, or any trade-off you've seen today."

POINTER: "And everything I've shown you — the code, the worksheets, the evidence — is in the GitHub repository on this slide."

[Hold for 2–3 seconds, then end the recording.]
