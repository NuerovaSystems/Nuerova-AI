# NuerovaAI

NuerovaAI is a tiny AI server I built from scratch.

It provides:

- `POST /ai/message` – combined endpoint with:
  - moderation (safe / unsafe)
  - 3-way sentiment (positive / neutral / negative) from a small PyTorch model
  - per-user memory (recent messages + mood summary)
  - simple chat replies
  - a calculator tool for basic math expressions

There is also a Raspberry Pi client (`nuerova_client.py`) that talks to the server over HTTP.

## Files

- `ai_api.py` – Flask API server (NuerovaAI brain)
- `train_sentiment.py` – trains the tiny sentiment model and saves `nuerova_sentiment.pt`
- `nuerova_client.py` – Raspberry Pi client that calls `/ai/message`
- `requirements.txt` – Python dependencies

## Quickstart (server)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# train sentiment model
python train_sentiment.py

# run API server
python ai_api.py
