# MIT RAG & Context Engineering — Capstone Project

This repository tracks my ongoing capstone project for the MIT program's
retrieval-augmented generation (RAG) / LLM course. The capstone is built
incrementally: each module contributes a checkpoint that extends the same
underlying retrieval-augmented LLM system, from a bare chatbot in Module 1
toward a full RAG pipeline by the end of the program.

## Repository organization

Each module's capstone checkpoint gets its own top-level folder, named
`Capstone Checkpoint <module>.<checkpoint>`. Inside each folder:

- **Solution file** — the completed code for that checkpoint.
- **Worksheet** — the lab/checkpoint worksheet or handout, if one was issued.
- A short `README.md` describing what the checkpoint covers and its status.

```
Capstone Checkpoint 1.1/
    lab_1_1_chatbot_solution.py
    WORKSHEET.md
    README.md
Capstone Checkpoint 2.1/
    README.md   (placeholder until this module is completed)
...
Capstone Checkpoint 8.1/
    README.md   (placeholder until this module is completed)
```

As each new module is completed, its checkpoint folder is filled in with the
solution file and worksheet, following this same pattern.

## Checkpoint status

| Checkpoint | Topic | Status |
|---|---|---|
| [1.1](./Capstone%20Checkpoint%201.1) | Chatbot with/without conversation memory | ✅ Complete |
| [2.1](./Capstone%20Checkpoint%202.1) | TBD | ⬜ Not started |
| [3.1](./Capstone%20Checkpoint%203.1) | TBD | ⬜ Not started |
| [4.1](./Capstone%20Checkpoint%204.1) | TBD | ⬜ Not started |
| [5.1](./Capstone%20Checkpoint%205.1) | TBD | ⬜ Not started |
| [6.1](./Capstone%20Checkpoint%206.1) | TBD | ⬜ Not started |
| [7.1](./Capstone%20Checkpoint%207.1) | TBD | ⬜ Not started |
| [8.1](./Capstone%20Checkpoint%208.1) | TBD | ⬜ Not started |

## Setup (per checkpoint)

Each checkpoint is a standalone Python script. General setup:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install python-dotenv langchain-openai langchain-core
```

Add a `.env` file (not committed — see `.gitignore`) with your OpenRouter API key:

```
OPENROUTER_API_KEY=sk-or-your-key-here
```
