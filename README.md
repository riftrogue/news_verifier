# AI News Verification System

A minimal, extendable pipeline using LangChain + Groq (ChatGroq), FAISS, and Tavily to verify short news claims and produce a single JSON verdict.

## Setup

1. Ensure you have Python 3.10+ and a virtual environment (optional).
2. Create a `.env` file with:

```
GROQ_API_KEY="your_groq_key"
TAVILY_API_KEY="your_tavily_key"
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

- Quick run with CLI input:

```bash
python main.py "Shahrukh Khan was born in West Bengal."
```

- Or put input into `data/temp_input.json`:

```json
{"news": "Shahrukh Khan was born in West Bengal."}
```

then run:

```bash
python main.py
```

Only the final JSON is printed. Intermediate agent steps are silent; logs are saved to `data/chat_history.json` and `data/verified_reports.json`.

## Notes
- FAISS index is stored in `data/embeddings`. If empty, the system falls back to web verification.
- You can pre-load domain facts into the vector DB by creating your own script to call `SimpleVectorDB.add([...])`.
- Trusted sources are configured in `configs/settings.yaml`.
