# Capstone Checkpoint 1.1 — Chatbot With/Without Memory

**Status:** Solution code complete. Worksheet added but not yet filled in.

## Contents
- `lab_1_1_chatbot_solution.py` — completed Lab 1.1 chatbot. Implements:
  - `build_no_history_respond()` — stateless chatbot (no memory across turns).
  - `build_history_respond()` — chatbot that carries full message history to the LLM.
  - A `USE_HISTORY` flag toggles between the two, and a CLI chat loop for manual testing.
- `Required_Capstone_Checkpoint_1_1_Worksheet.docx` — official checkpoint worksheet,
  "Evaluating When Retrieval Is Required." This is the blank assignment template as
  issued; the response sections (scenario selection, prompt/output evidence, limitation
  analysis, retrieval justification, reflection) still need to be filled in and
  resubmitted through the course platform.

## How to run
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install python-dotenv langchain-openai langchain-core
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
python "lab_1_1_chatbot_solution.py"
```
