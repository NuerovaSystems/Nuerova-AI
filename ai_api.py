from os import name
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import re

app = Flask(__name__)

# Simple in-memory store for user state (resets when server restarts)
memory = {}
# ===== NuerovaAI tiny sentiment model setup =====

class TinySentimentModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return torch.sigmoid(logits)


# load model + vocab at startup
sentiment_checkpoint = torch.load("nuerova_sentiment.pt")
word2idx = sentiment_checkpoint["word2idx"]

sentiment_model = TinySentimentModel(input_dim=len(word2idx), hidden_dim=16)
sentiment_model.load_state_dict(sentiment_checkpoint["model_state_dict"])
sentiment_model.eval()

def text_to_vector(text: str) -> torch.Tensor:
    vec = torch.zeros(len(word2idx))
    for word in text.lower().split():
        if word in word2idx:
            vec[word2idx[word]] += 1
    return vec

def make_error(message, status_code):
    return jsonify({
        "error": True,
        "message": message
    }), status_code

def try_calculate(text: str):
    """
    Try to evaluate a simple math expression from the text.
    Returns (success: bool, result_or_message: str)
    """
    # Extract something that looks like a math expression
    # e.g. "2+2", "5 * 7", "10 / 3"
    match = re.search(r"[\d\s\+\-\*\/\.\(\)]+", text)
    if not match:
        return False, "I couldn't find a clear math expression to calculate."

    expr = match.group(0).strip()

    # Very basic safety: only allow digits, spaces, + - * / . ( )
    if not re.fullmatch(r"[\d\s\+\-\*\/\.\(\)]+", expr):
        return False, "The expression looks unsafe to evaluate."

    try:
        result = eval(expr, {"__builtins__": {}})
    except Exception:
        return False, f"I couldn't evaluate the expression: {expr!r}."

    return True, f"The result of {expr} is {result}."

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/ai/complete", methods=["POST"])
def ai_complete():
    # 1. Check Content-Type
    if not request.is_json:
        return make_error("Request body must be JSON", 400)

    data = request.get_json()

    # NEW: check that we actually got a dict
    if data is None or not isinstance(data, dict):
        return make_error("Request body must be a JSON object", 400)

    # 2. Presence check
    if "prompt" not in data:
        return make_error("Field 'prompt' is required", 400)

    prompt = data["prompt"]

    # 3. Type check
    if not isinstance(prompt, str):
        return make_error("Field 'prompt' must be a string", 400)

        # 4. Tiny rule-based "AI brain"
    text = prompt.lower().strip()

    if "hello" in text or "hi" in text:
        result = "Hi there, I'm your mini AI. How can I help you today?"
    elif "who are you" in text:
        result = "I'm a tiny AI running inside your Flask server."
    elif "help" in text:
        result = "I can respond to your messages with simple, hard-coded rules for now."
    elif "bye" in text or "goodbye" in text:
        result = "Goodbye! Talk to you later."
    else:
        result = f"You said: {prompt}. I don't know much yet, but I'm learning."

    return jsonify({
        "result": result
    }), 200

@app.route("/ai/sentiment", methods=["POST"])
def ai_sentiment():
    # 1. Validate JSON
    if not request.is_json:
        return make_error("Request body must be JSON", 400)

    data = request.get_json()

    if data is None or not isinstance(data, dict):
        return make_error("Request body must be a JSON object", 400)

    # 2. Presence + type checks
    if "text" not in data:
        return make_error("Field 'text' is required", 400)

    text = data["text"]

    if not isinstance(text, str):
        return make_error("Field 'text' must be a string", 400)

    # 3. Run through NuerovaAI sentiment model
    vec = text_to_vector(text).unsqueeze(0)  # shape [1, vocab_size]

    with torch.no_grad():
        prob = sentiment_model(vec).item()  # between 0 and 1

        label = "positive" if prob >= 0.5 else "negative"

    # tiny "reaction" logic, like a robot mood
    if label == "positive":
        reaction = "NuerovaAI: I'm glad you feel good about this. 🙂"
    else:
        reaction = "NuerovaAI: I'm sorry this feels bad. How could it be improved?"

    return jsonify({
        "model": "NuerovaAI tiny sentiment v1",
        "text": text,
        "label": label,
        "score": prob,
        "reaction": reaction
    }), 200

@app.route("/ai/moderate", methods=["POST"])
def ai_moderate():
    # 1. Validate JSON
    if not request.is_json:
        return make_error("Request body must be JSON", 400)

    data = request.get_json()

    if data is None or not isinstance(data, dict):
        return make_error("Request body must be a JSON object", 400)

    # 2. Presence + type checks
    if "text" not in data:
        return make_error("Field 'text' is required", 400)

    text = data["text"]

    if not isinstance(text, str):
        return make_error("Field 'text' must be a string", 400)

    # 3. Tiny rule-based moderation "brain"
    lowered = text.lower()

    unsafe_keywords = [
        "kill",
        "hurt myself",
        "hurt someone",
        "bomb",
        "threat",
        "hate crime",
        "suicide",
        "self harm",
    ]

    is_safe = True
    matched = []

    for word in unsafe_keywords:
        if word in lowered:
            is_safe = False
            matched.append(word)

    label = "safe" if is_safe else "unsafe"

    return jsonify({
        "model": "NuerovaAI tiny moderator v1",
        "text": text,
        "label": label,
        "unsafe_matches": matched
    }), 200

@app.route("/ai/chat", methods=["POST"])
def ai_chat():
    # 1. Validate JSON
    if not request.is_json:
        return make_error("Request body must be JSON", 400)

    data = request.get_json()

    if data is None or not isinstance(data, dict):
        return make_error("Request body must be a JSON object", 400)

    # 2. Presence + type checks
    if "messages" not in data:
        return make_error("Field 'messages' is required", 400)

    messages = data["messages"]

    if not isinstance(messages, list):
        return make_error("Field 'messages' must be a list", 400)

    if len(messages) == 0:
        return make_error("Field 'messages' cannot be empty", 400)

    # 3. Get last user message
    last_user_msg = None
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            last_user_msg = msg["content"]
            break

    if last_user_msg is None:
        return make_error("At least one user message with 'role': 'user' and 'content' string is required", 400)

    text = last_user_msg.lower().strip()

    # 4. Tiny chat "brain"
    if "hello" in text or "hi" in text:
        reply = "Hi, I'm NuerovaAI. What would you like to talk about?"
    elif "joke" in text:
        reply = "Here's a lame joke: Why did the robot cross the road? Because it was programmed by a human. 😄"
    elif "who are you" in text:
        reply = "I'm a tiny chat AI you built, running inside your Flask server."
    elif "sad" in text or "upset" in text:
        reply = "I'm sorry you're feeling that way. Do you want to talk about it?"
    else:
        reply = f"You said: {last_user_msg}. I'm still a simple NuerovaAI, but I'm learning."

    return jsonify({
        "model": "NuerovaAI tiny chat v1",
        "reply": reply
    }), 200

@app.route("/ai/message", methods=["POST"])
def ai_message():
    # 1. Validate JSON
    if not request.is_json:
        return make_error("Request body must be JSON", 400)

    data = request.get_json()

    if data is None or not isinstance(data, dict):
        return make_error("Request body must be a JSON object", 400)

    # require text and user_id
    if "text" not in data:
        return make_error("Field 'text' is required", 400)
    if "user_id" not in data:
        return make_error("Field 'user_id' is required", 400)

    text = data["text"]
    user_id = data["user_id"]

    if not isinstance(text, str):
        return make_error("Field 'text' must be a string", 400)
    if not isinstance(user_id, str):
        return make_error("Field 'user_id' must be a string", 400)

    lowered = text.lower().strip()

    # --- Moderation step (same as before) ---
    unsafe_keywords = [
        "kill",
        "hurt myself",
        "hurt someone",
        "bomb",
        "threat",
        "hate crime",
        "suicide",
        "self harm",
    ]

    is_safe = True
    matched = []

    for word in unsafe_keywords:
        if word in lowered:
            is_safe = False
            matched.append(word)

    moderation_label = "safe" if is_safe else "unsafe"

    # If unsafe, store it in memory and stop here
    if not is_safe:
        # update memory
        user_state = memory.get(user_id, {
            "messages": [],
            "last_sentiment": None,
            "unsafe_count": 0
        })
            # If unsafe, store it in memory and stop here
    if not is_safe:
        # update memory
        user_state = memory.get(user_id, {
            "messages": [],
            "last_sentiment": None,
            "unsafe_count": 0,
            "mood_summary": {"positive": 0, "negative": 0, "neutral": 0}
        })
        user_state["messages"].append({"text": text, "moderation": "unsafe"})
        user_state["unsafe_count"] += 1

        # leave mood_summary as-is for unsafe messages
        memory[user_id] = user_state

        return jsonify({
            "moderation": {
                "label": moderation_label,
                "unsafe_matches": matched
            },
            "sentiment": None,
            "chat": {
                "reply": (
                    "NuerovaAI: I can't respond to this request. "
                    "If you're in danger or thinking of self-harm, please reach out "
                    "to a trusted person or professional."
                )
            },
            "memory": {
                "unsafe_count": user_state["unsafe_count"],
                "mood_summary": user_state.get("mood_summary", {})
            }
        }), 200

           # --- Sentiment step (reuse your model) ---
    vec = text_to_vector(text).unsqueeze(0)

    with torch.no_grad():
        prob = sentiment_model(vec).item()

    if prob <= 0.3:
        sentiment_label = "negative"
    elif prob >= 0.7:
        sentiment_label = "positive"
    else:
        sentiment_label = "neutral"

    # --- Update memory for safe message ---
    user_state = memory.get(user_id, {
        "messages": [],
        "last_sentiment": None,
        "unsafe_count": 0,
        "mood_summary": {"positive": 0, "negative": 0, "neutral": 0}
    })
    user_state["messages"].append({"text": text, "sentiment": sentiment_label})
    user_state["last_sentiment"] = sentiment_label
    # keep only last 10 messages for this user
    user_state["messages"] = user_state["messages"][-10:]

    # recompute mood_summary from recent messages
    history = user_state["messages"]
    recent = history[-5:]  # last 5 messages

    recent_pos = sum(1 for m in recent if m.get("sentiment") == "positive")
    recent_neg = sum(1 for m in recent if m.get("sentiment") == "negative")
    recent_neu = sum(1 for m in recent if m.get("sentiment") == "neutral")

    user_state["mood_summary"] = {
        "positive": recent_pos,
        "negative": recent_neg,
        "neutral": recent_neu
    }

    memory[user_id] = user_state

    mood = user_state["mood_summary"]
    recent_neg = mood["negative"]
    recent_pos = mood["positive"]

    # --- Simple calculator tool ---
    # If the user seems to be asking for a calculation, try it
    if any(word in lowered for word in ["calculate", "calc", "what is", "what's"]) and any(ch.isdigit() for ch in text):
        ok, calc_message = try_calculate(text)
        if ok:
            reply = f"{calc_message} (calculated by NuerovaAI)."
        else:
            reply = calc_message

        return jsonify({
            "moderation": {
                "label": moderation_label,
                "unsafe_matches": matched
            },
            "sentiment": {
                "label": sentiment_label,
                "score": prob
            },
            "chat": {
                "reply": reply
            },
            "memory": {
                "last_sentiment": user_state["last_sentiment"],
                "unsafe_count": user_state["unsafe_count"],
                "message_count": len(user_state["messages"]),
                "mood_summary": user_state["mood_summary"]
            }
        }), 200

    # --- Chat reply step (use sentiment + history) ---
    if "hello" in lowered or "hi" in lowered:
        if recent_neg >= 3:
            reply = (
                "Hi again. I've seen several negative messages from you recently. "
                "Do you want to talk about what's been going wrong?"
            )
        elif recent_pos >= 3:
            reply = (
                "Hi again! You've sounded pretty positive lately — that's great. "
                "What's been going well?"
            )
        elif user_state["last_sentiment"] == "negative":
            reply = "Hi again. I've noticed some of your messages sound negative. Want to talk about it?"
        elif user_state["last_sentiment"] == "positive":
            reply = "Hi again! You seemed positive in your last message — I love that. How are you now?"
        else:  # neutral
            reply = "Hi, I'm NuerovaAI. How are you feeling today?"

    elif sentiment_label == "positive":
        if recent_pos >= 3:
            reply = "You're on a roll with positive messages — I love this energy!"
        else:
            reply = "I'm glad you're feeling positive about this!"

    elif sentiment_label == "negative":
        if recent_neg >= 3:
            reply = (
                "It seems like you've been negative in several recent messages. "
                "Do you want to focus on the biggest issue first?"
            )
        else:
            reply = "I'm sorry this feels negative. Want to talk more about what's bothering you?"

    else:  # neutral
        reply = "Sounds like you're feeling kind of neutral about this."

    return jsonify({
        "moderation": {
            "label": moderation_label,
            "unsafe_matches": matched
        },
        "sentiment": {
            "label": sentiment_label,
            "score": prob
        },
        "chat": {
            "reply": reply
        },
        "memory": {
            "last_sentiment": user_state["last_sentiment"],
            "unsafe_count": user_state["unsafe_count"],
            "message_count": len(user_state["messages"]),
            "mood_summary": user_state["mood_summary"]
        }
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)