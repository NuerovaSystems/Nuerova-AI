import requests

SERVER_URL = os.getenv("NUEROVAAI_SERVER_URL", "http://127.0.0.1:5001/ai/message" # Your Mac's NuerovaAI server

def send_to_nuerova(text, user_id="pi_user_1"):
    payload = {"user_id": user_id, "text": text}
    r = requests.post(SERVER_URL, json=payload, timeout=10)
    r.raise_for_status()
    data = r.json()
    reply = data["chat"]["reply"]
    sentiment = None
    mood = None

    if data.get("sentiment"):
        sentiment = data["sentiment"]["label"]
    if data.get("memory") and data["memory"].get("mood_summary"):
        mood = data["memory"]["mood_summary"]

    return reply, sentiment, mood

if __name__ == "__main__":
    print("Type messages for NuerovaAI (Pi client). Type 'quit' to exit.")
    while True:
        user_text = input("You: ")
        if user_text.lower().strip() in ("quit", "exit"):
            break

        try:
            reply, sentiment, mood = send_to_nuerova(user_text)
            print("NuerovaAI:", reply)
            print("sentiment:", sentiment)
            print("mood:", mood)
        except Exception as e:
            print("Error talking to NuerovaAI:", e)
