from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pickle
import numpy as np
import cv2
import os
import mediapipe as mp
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------- SIGN → TEXT ---------------- #
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file found"})

    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        row = []
        for lm in handLms.landmark:
            row.extend([lm.x, lm.y, lm.z])
        features = np.array(row).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": str(prediction[0])})
    else:
        return jsonify({"error": "No hand detected"})

# ---------------- TEXT → SIGN ---------------- #
# Letters
sign_dict = {chr(i): f"signs/{chr(i)}.png" for i in range(65, 91)}
# Punctuation
# Add punctuation mapping
# Mapping punctuation → images
punct_dict = {
    ".": "signs/fullstop.png",
    ",": "signs/comma.png",
    "!": "signs/exclamation.png",
    "?": "signs/question.png"
}

@app.route("/text-to-sign", methods=["GET", "POST"])
def text_to_sign():
    if request.method == "POST":
        text = request.form.get("text", "").upper()
        words_signs = []  # list of words, each word = list of image paths
        word = []

        for ch in text:
            if ch == " ":
                if word:
                    words_signs.append(word)
                    word = []
            elif ch in sign_dict:
                word.append(sign_dict[ch])
            elif ch in punct_dict:
                word.append(punct_dict[ch])

        # Add last word
        if word:
            words_signs.append(word)

        session["words_signs"] = words_signs
        session["text"] = text
        return redirect(url_for("text_to_sign"))

    words_signs = session.get("words_signs", [])
    text = session.get("text", "")
    return render_template("text_to_sign.html", words_signs=words_signs, text=text)


# ---------------- SIGN → TEXT PAGE ---------------- #
@app.route("/sign-to-text")
def sign_to_text():
    return render_template("sign_to_text.html")

# ---------------- HOME ---------------- #
@app.route("/")
def home():
    return render_template("home.html")

# ---------------- RUN APP ---------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
