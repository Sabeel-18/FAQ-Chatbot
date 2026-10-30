"""Simple Flask chat UI.  Run:  python app.py  then open http://127.0.0.1:5000"""
from flask import Flask, jsonify, render_template, request

from chatbot import FAQChatbot

app = Flask(__name__)
bot = FAQChatbot()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    message = (request.get_json(silent=True) or {}).get("message", "")
    answer, matched, score = bot.get_response(message)
    return jsonify({"answer": answer, "matched": matched, "score": round(score, 2)})


if __name__ == "__main__":
    app.run(debug=True)
