from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os, json, re
from datetime import datetime
from werkzeug.utils import secure_filename
from modules.preprocessing import train_classifiers, predict_intent, save_training_data, load_training_data
from modules.file_qa import extract_text, answer_from_text
from modules.memory import add_memory, get_memories
from modules.continuous_learning import add_feedback

app = Flask(__name__)
app.secret_key = "change-this-secret-key"

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ADMIN_USER = "admin"
ADMIN_PASSWORD = "admin123"

@app.route("/")
def index():
    return render_template("index.html", memories=get_memories()[-5:])

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("username") == ADMIN_USER and request.form.get("password") == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect(url_for("admin"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect(url_for("index"))

@app.route("/admin")
def admin():
    if not session.get("admin"):
        return redirect(url_for("login"))
    data = load_training_data()
    return render_template("admin.html", data=data, memories=get_memories())

@app.route("/train", methods=["POST"])
def train():
    if not session.get("admin"):
        return jsonify({"error": "Admin login required"}), 403
    text = request.form.get("text", "").strip()
    label = request.form.get("label", "").strip()
    if not text or not label:
        flash("Text and label are required.", "error")
        return redirect(url_for("admin"))
    save_training_data(text, label)
    result = train_classifiers()
    flash(f"Training complete. Best model by F1: {result['best_model']}", "success")
    return redirect(url_for("admin"))

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    if not question:
        flash("Please enter a question.", "error")
        return redirect(url_for("index"))

    prediction = predict_intent(question)
    answer = (
        f"I understood your request as: {prediction['label']}. "
        f"This demo uses a trained text classifier and memory. "
        f"For a real LLM, connect an approved LLM API or local model."
    )
    add_memory(question, answer, prediction["label"])
    return render_template("index.html", answer=answer, prediction=prediction,
                           memories=get_memories()[-5:])

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    question = request.form.get("file_question", "").strip()
    if not file or not file.filename:
        flash("Select a file first.", "error")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    try:
        text = extract_text(path)
        answer = answer_from_text(text, question) if question else (
            "File uploaded successfully. Ask a question about the file."
        )
        add_memory(question or "File upload", answer, "file_qa")
        return render_template("index.html", answer=answer, file_name=filename,
                               memories=get_memories()[-5:])
    except Exception as exc:
        flash(f"Could not process file: {exc}", "error")
        return redirect(url_for("index"))

@app.route("/feedback", methods=["POST"])
def feedback():
    question = request.form.get("question", "")
    answer = request.form.get("answer", "")
    value = request.form.get("value", "")
    add_feedback(question, answer, value)
    return jsonify({"ok": True})

if __name__ == "__main__":
    if not os.path.exists(os.path.join(DATA_DIR, "training_data.json")):
        save_training_data("hello", "greeting")
        save_training_data("what is machine learning", "machine_learning")
        save_training_data("explain artificial intelligence", "artificial_intelligence")
        save_training_data("help me with my project", "project_help")
    if not os.path.exists(os.path.join(DATA_DIR, "model_metrics.json")):
        train_classifiers()
    app.run(debug=True, host="127.0.0.1", port=5000)
