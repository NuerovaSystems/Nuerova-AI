# NuerovaAI

NuerovaAI is a tiny AI server I built from scratch.

It provides:

- `POST /ai/message` – combined endpoint with:
  - moderation (safe / unsafe)
  - 3-way sentiment (positive / neutral / negative) from a small PyTorch model
  - per-user memory (recent messages + mood summary)
  - simple chat replies
  - a calculator tool for basic math expressions
- `POST /ai/sentiment` – run text through the tiny sentiment model
- `POST /ai/moderate` – simple rule-based content moderation
- `POST /ai/chat` – tiny chat endpoint using a `messages` array (OpenAI-style)
- `POST /ai/complete` – tiny completion-style endpoint using a single `prompt`
- `GET /health` – basic health check

There is also a Raspberry Pi client (`nuerova_client.py`) that talks to the server over HTTP.

---

## Files

- `ai_api.py` – Flask API server (NuerovaAI brain)
- `train_sentiment.py` – trains the tiny sentiment model and saves `nuerova_sentiment.pt`
- `nuerova_client.py` – Raspberry Pi client that calls `/ai/message`
- `requirements.txt` – Python dependencies
- `README.md` – this file

---

## Quickstart (server)

```bash
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
# .venv\Scripts\activate   # on Windows

pip install -r requirements.txt

# train sentiment model
python train_sentiment.py

# run API server
python ai_api.py
