from flask import Flask, render_template, request
from chatbot import get_response   # Import function from chatbot.py

app = Flask(__name__)

# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Chatbot reply API
@app.route("/get")
def chatbot_reply():
    msg = request.args.get("msg")   # Get user message
    response = get_response(msg)   # Get bot reply
    return response


# Run Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
