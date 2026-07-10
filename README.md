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

By default the server runs on http://127.0.0.1:5001.
API Documentation (Swagger) 
Interactive API docs are available via Swagger UI:
 
http://127.0.0.1:5001/apidocs
 
From there you can:
 
See all endpoints and their request/response shapes
Use “Try it out” to call the live API from your browser
Endpoints 
GET /health
 
Simple health check.
 
Response 200
Json
1{
2  "status": "ok"
3}
POST /ai/message
 
Combined endpoint that does moderation, sentiment, memory, and a reply.
 
Request body
Json
{
  "text": "I'm really excited about this new project!",
  "user_id": "user_123"
}
Success 200 (example)
Json
{
  "moderation": {
    "label": "safe",
    "unsafe_matches": []
},
  "sentiment": {
    "label": "positive",
    "score": 0.95
},
  "chat": {
    "reply": "Hi again! You seemed positive in your last message — I love that. How are you now?"
  },
  "memory": {
    "unsafe_count": 0,
    "mood_summary": {
      "positive": 1,
      "negative": 0,
      "neutral": 0
},
    "last_sentiment": "positive",
    "message_count": 1
  }
}
Error 400 (example)
Json
{
  "error": true,
  "message": "Field 'text' is required"
}
POST /ai/sentiment
 
Analyze sentiment using the tiny PyTorch model.
 
Request
Json
{
  "text": "I really love working on this project."
}
Success 200 (example)
Json
{
  "model": "NuerovaAI tiny sentiment v1",
  "text": "I really love working on this project.",
  "label": "positive",
  "score": 0.99,
  "reaction": "NuerovaAI: I'm glad you feel good about this. 🙂"
}
Error 400 (example)
Json
{
  "error": true,
  "message": "Field 'text' is required"
}
POST /ai/moderate
 
Simple rule-based content moderation.
 
Request
Json
{
  "text": "I want to hurt myself."
}
Success 200 (example)
Json
{
  "model": "NuerovaAI tiny moderator v1",
  "text": "I want to hurt myself.",
  "label": "unsafe",
  "unsafe_matches": ["hurt myself"]
}
Error 400 (example)
Json
{
  "error": true,
  "message": "Field 'text' is required"
}
POST /ai/chat
 
Tiny chat endpoint using an OpenAI-style messages array.
 
Request
Json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "hello, who are you?" }
  ]
}
Success 200 (example)
Json
{
  "model": "NuerovaAI tiny chat v1",
  "reply": "Hi, I'm NuerovaAI. What would you like to talk about?"
}
Error 400 (example)
Json
{
  "error": true,
  "message": "Field 'messages' is required"
}
POST /ai/complete
 
Simple completion-style endpoint using a single prompt.
 
Request
Json
{
  "prompt": "hello"
}
Success 200 (example)
Json
{
  "result": "Hi there, I'm your mini AI. How can I help you today?"
}
Error 400 (example)
Json
{
  "error": true,
  "message": "Field 'prompt' is required"
}
Error Format 
All validation errors use a consistent JSON envelope:
Json
{
  "error": true,
  "message": "Field 'text' is required"
}
The message field will change depending on which validation failed (missing field, wrong type, etc.).
Raspberry Pi Client 
The nuerova_client.py script is a simple client that sends messages to /ai/message from a Raspberry Pi over HTTP.
 
