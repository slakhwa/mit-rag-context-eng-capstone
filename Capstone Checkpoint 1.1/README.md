# Capstone Checkpoint 1.1 — Chatbot With/Without Memory

**Status:** Solution complete. Worksheet not yet added.

## Contents
- `lab_1_1_chatbot_solution.py` — completed Lab 1.1 chatbot. Implements:
  - `build_no_history_respond()` — stateless chatbot (no memory across turns).
  - `build_history_respond()` — chatbot that carries full message history to the LLM.
  - A `USE_HISTORY` flag toggles between the two, and a CLI chat loop for manual testing.
- `WORKSHEET.md` — placeholder. The lab worksheet/handout wasn't provided as a separate
  file for this checkpoint; add it here if/when it's issued.

## How to run
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install python-dotenv langchain-openai langchain-core
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
python "lab_1_1_chatbot_solution.py"
```
