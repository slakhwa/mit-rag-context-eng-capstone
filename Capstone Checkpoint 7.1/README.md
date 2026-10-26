# Capstone Checkpoint 7.1 — Production Readiness: Security and Performance Optimization

**Status:** ✅ Complete — sanitizer extended and verified against all 3 real Checkpoint 6.1 probes, cache/router cost wins measured, mixed (not clean) results reported honestly, worksheet filled in.

**Scenario (same as 1.1–6.1): Research Paper Navigator.**

This checkpoint fixes the gaps Checkpoint 6.1's audit found: "there is no
perfect fix — stack multiple layers of defense."

## What was fixed, and how it was verified before trusting it

Before changing anything, I checked whether the starter's own sanitizer
regexes actually caught my 3 real Checkpoint 6.1 attacks. They caught only
**1 of 3** — `command_injection` matched the roleplay-focused trigger list,
but `context_poisoning` (used `"BEGIN PAPER EXCERPT"`, not literally `"email
block"`) and `retrieval_query_pollution` (used `"system note"`/`"operator
override"` wording never in the trigger list) both slipped through.

**Fixes made:**
1. Generalized `_EMAIL_BLOCK_RE` from the literal phrase `"email block"` to
   any `"BEGIN/END <words> BLOCK|EXCERPT|DOCUMENT|SOURCE"` marker.
2. Added directive-injection phrasing (`"system note"`, `"operator override"`,
   `"mandatory compliance"`, `"higher priority than"`, `"compliance check"`)
   to `_INJECTION_RE`.
3. Wired sanitization into the **full agent loop** (`agentic_answer`) and the
   **direct path** (`answer_direct`), not just the standalone probe demo —
   Checkpoint 6.1 found the planner (`decide()`) had zero trust boundary at
   all and literally echoed an injected directive in its own reasoning.
4. Switched both paths to the hardened XML trust boundary
   (`HARDENED_XML_SYSTEM` + `<documents>`/`<user_question>` tags).

I verified the extended regexes against 5 legitimate research questions
(including ones using "override," "context," "block" naturally) before
trusting them — no false positives.

## Results — genuinely mixed, not a clean win

- **Regex coverage: 1/3 → 3/3**, verified directly (pure regex, no LLM calls).
- **Single-call hardened path** (`probe_agent`/`answer_direct`): all 3 probes
  produced a safe answer, raw and sanitized — this held regardless of the
  sanitizer change, because `ANSWER_SYSTEM`'s own grounding instruction is an
  independent second line of defense.
- **But the full multistep agent loop told a different story** on
  `retrieval_query_pollution`: sanitizing did **not** restore correct
  retrieval — the clean question retrieves the correct Bowers/Stitch chunks,
  but the *sanitized* attack retrieves a **third, still-wrong** set of chunks
  (different from the raw attack's wrong chunks). A phrase blocklist removes
  specific trigger words but leaves ~30 words of generic filler
  ("search queries," "information," "executed"...) that still dilute BM25's
  keyword match — a different failure mode a blocklist can't fix.
- **The sanitized full-loop run also cost 65% more tokens** (13,851 vs.
  8,401) **and gave a less cautious answer** than the raw, unsanitized run —
  it conflated "Ellis et al. 2021" content into an answer about the
  specifically-named 2018 EC2 paper, while the raw run correctly said it
  lacked that information. **Sanitizing the input didn't fix, and in this run
  didn't even contain, the Checkpoint 5.1/6.1 citation-misattribution bug.**

**Cost wins were clean, by contrast:** a cache hit saved 100% of tokens and
latency on a repeated question (2,704 tokens/3,471 ms → 0/0.1 ms), and the
router sends simple one-hop questions to a 1-call path instead of the agent's
2–4 calls.

## Contents

- **`Required_Capstone_Checkpoint_7_1_Worksheet_completed.docx`** — the
  graded deliverable with the full, honest mixed-results writeup above.
- **`capstone_checkpoint_7_1_production_ready_starter.py`** — adapted to the
  real corpus/agent, with `my_production_plan()` filled in, the sanitizer
  regexes extended and verified, sanitization wired into the full agent loop
  (not just the probe demo), and a `regex_coverage_before_after()` +
  retrieval-layer + planner-level comparison added beyond the original
  starter's single toy-corpus demo.
- **`checkpoint_7_1_agent.log`** — full evidence: cost measurements, regex
  coverage per probe, the clean/raw/sanitized retrieval comparison, all 3
  probes' answers at both the single-call and full-loop level.

**Not included** (regenerate locally, see Checkpoint 2.1's README):
`paper_text/` — same corpus used throughout, kept out of git.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install python-dotenv langchain-openai langchain-core rank-bm25
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
# paper_text/ is NOT included (gitignored); regenerate per Checkpoint 2.1's README.
python capstone_checkpoint_7_1_production_ready_starter.py
```
