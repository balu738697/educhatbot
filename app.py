from flask import Flask, render_template, request, jsonify
from groq import Groq
import json, os, numpy as np
from datetime import datetime

app = Flask(__name__)

client = Groq(api_key="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")

HISTORY_FILE = "chat_history.json"
USERS_FILE   = "users.json"
THRESHOLD    = 0.45

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/history", methods=["GET"])
def get_history():
    user_id = request.args.get("user_id")
    history = load_json(HISTORY_FILE, [])
    if user_id:
        history = [h for h in history if h.get("user_id") == user_id]
    return jsonify(history)

@app.route("/clear_session", methods=["POST"])
def clear_session():
    data       = request.json
    session_id = data.get("session_id")
    history    = load_json(HISTORY_FILE, [])
    history    = [h for h in history if h.get("session_id") != session_id]
    save_json(HISTORY_FILE, history)
    return jsonify({"status": "cleared"})

@app.route("/clear_all", methods=["POST"])
def clear_all():
    data    = request.json
    user_id = data.get("user_id")
    history = load_json(HISTORY_FILE, [])
    history = [h for h in history if h.get("user_id") != user_id]
    save_json(HISTORY_FILE, history)
    return jsonify({"status": "cleared"})

@app.route("/verify_face", methods=["POST"])
def verify_face():
    data       = request.json
    descriptor = data.get("descriptor")
    if not descriptor:
        return jsonify({"status": "error", "message": "No descriptor provided"})

    users     = load_json(USERS_FILE, [])
    new_desc  = np.array(descriptor)
    best_match = None
    best_dist  = float("inf")

    for user in users:
        saved_desc = np.array(user["descriptor"])
        dist = float(np.linalg.norm(new_desc - saved_desc))
        if dist < best_dist:
            best_dist  = dist
            best_match = user

    if best_match and best_dist < THRESHOLD:
        return jsonify({
            "status":     "known",
            "user_id":    best_match["user_id"],
            "user_name":  best_match.get("name", "User"),
            "created_at": best_match.get("created_at", ""),
            "distance":   round(best_dist, 4)
        })
    else:
        user_id    = "user_" + datetime.now().strftime("%Y%m%d%H%M%S%f")
        user_count = len(users) + 1
        new_user   = {
            "user_id":    user_id,
            "name":       f"User {user_count}",
            "descriptor": descriptor,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        users.append(new_user)
        save_json(USERS_FILE, users)
        return jsonify({
            "status":    "new",
            "user_id":   user_id,
            "user_name": f"User {user_count}"
        })

@app.route("/chat", methods=["POST"])
def chat():
    data         = request.json
    user_message = data.get("message", "").lower()
    session_id   = data.get("session_id", "default")
    user_id      = data.get("user_id", "unknown")

    education_keywords = [
        "education", "school", "college", "university", "student", "teacher",
        "study", "learn", "exam", "test", "homework", "assignment", "class",
        "subject", "math", "science", "history", "geography", "english",
        "physics", "chemistry", "biology", "lesson", "syllabus", "degree",
        "course", "lecture", "textbook", "grade", "marks", "scholarship",
        "training", "knowledge", "skill", "reading", "writing", "arithmetic",
        "what is", "explain", "define", "how does", "why is", "who is",
        "formula", "theorem", "equation", "element", "atom", "cell",
        "grammar", "literature", "poem", "essay", "language", "culture"
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history   = load_json(HISTORY_FILE, [])

    if not any(word in user_message for word in education_keywords):
        error_reply = "⚠️ Out of Domain: I am an Education chatbot. Please ask questions related to subjects like Mathematics, Science, History, Geography, English, or any academic topic."
        history.append({"user_id": user_id, "session_id": session_id, "datetime": timestamp, "user": user_message, "bot": error_reply, "type": "error"})
        save_json(HISTORY_FILE, history)
        return jsonify({"reply": error_reply, "type": "error"})

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are EduBot, an expert education assistant. Answer questions clearly and in a well-structured format.
Rules:
- Use numbered points for steps or lists
- Use bullet points for details
- Use headings where needed
- Keep each point short and clear
- State definition first if applicable
- End with a summary if the answer is long"""
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            max_tokens=1024,
            temperature=0.7
        )

        bot_reply = completion.choices[0].message.content
        history.append({"user_id": user_id, "session_id": session_id, "datetime": timestamp, "user": user_message, "bot": bot_reply, "type": "success"})
        save_json(HISTORY_FILE, history)
        return jsonify({"reply": bot_reply, "type": "success"})

    except Exception as e:
        app.logger.exception("Groq error")
        return jsonify({"reply": f"⚠️ API Error: {str(e)}", "type": "error"}), 200

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
