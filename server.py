import json
import os
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_FILE = os.path.join(BASE_DIR, "signals.json")


@app.route("/")
def dashboard():
    return send_from_directory(BASE_DIR, "dashboard.html")


@app.route("/signals")
def signals():
    try:
        with open(SIGNALS_FILE, "r") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"last_updated": None, "counts": {"BUY": 0, "SELL": 0, "ALERT": 0}, "events": []})


if __name__ == "__main__":
    print("[*] Dashboard running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
