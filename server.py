from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pickle
import numpy as np
import cv2
import os
import nltk
import mediapipe as mp
from flask_cors import CORS
from nltk.corpus import words

# Download word corpus (first time only)
nltk.download("words", quiet=True)

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
# English dictionary
valid_words = set(words.words())

# Letter signs A–Z
sign_dict = {chr(i): f"signs/{chr(i)}.png" for i in range(65, 91)}

# Number signs 0–9
number_dict = {str(i): f"signs/{i}.png" for i in range(10)}

# Punctuation signs
punct_dict = {
    ".": "signs/fullstop.png",
    ",": "signs/comma.png",
    "!": "signs/exclamation.png",
    "?": "signs/question.png"
}

space_sign = "signs/space.png"


# ✅ FULL WORD → SIGN VIDEO (YouTube embed)
word_video_dict = {
    "AEROPLANE": "https://www.youtube.com/embed/6M1rP2r672o",
    "TRAIN": "https://www.youtube.com/embed/6ZaYY0vo6pk",
    "TICKET": "https://www.youtube.com/embed/nBqeyGg8Kvs",
    "COMPUTER":"https://www.youtube.com/embed/5IDn1RRczUo",
    "TELEVISION": "https://www.youtube.com/embed/ids6uU-p-lk",
  "CAMERA": "https://www.youtube.com/embed/xY5tQfR7BiI",
  "RADIO": "https://www.youtube.com/embed/_BPtLQumNF4",
  "BOOK": "https://www.youtube.com/embed/l2ytof4d1r0",
  "BOTTLE": "https://www.youtube.com/embed/zk-I0DPD4gk",
  "TIFFIN": "https://www.youtube.com/embed/YT1x1wgyjLQ",
  "APPLE": "https://www.youtube.com/embed/Lu_R01JQky8",
  "MANGO": "https://www.youtube.com/embed/fcuSg0jzcwI",
  "DOG": "https://www.youtube.com/embed/r9QQ0jQ64IQ",
  "BIRD": "https://www.youtube.com/embed/ThLhLlan5Yo",
    # add your own words here
}


@app.route("/text-to-sign", methods=["GET", "POST"])
def text_to_sign():

    if request.method == "POST":
        text = request.form.get("text", "")
        tokens = text.split()

        words_signs = []
        invalid_words = []
        i = 0

        while i < len(tokens):
            w = tokens[i]

            # ✅ Word-level sign video support
            word_clean = ''.join([ch for ch in w if ch.isalpha()])
            UPPER_WORD = word_clean.upper()

            if UPPER_WORD in word_video_dict:
                words_signs.append({
                    "type": "word_video",
                    "word": UPPER_WORD,
                    "video": word_video_dict[UPPER_WORD]
                })
                i += 1
                continue

            # ✅ Normal letters fallback
            word_clean = ''.join([ch for ch in w if ch.isalpha()])
            digits_only = ''.join([ch for ch in w if ch.isdigit()])
            punct_only = ''.join([ch for ch in w if ch in punct_dict])

            if (word_clean.lower() in valid_words) or digits_only or punct_only:
                signs = []
                for ch in w.upper():
                    if ch in sign_dict:
                        signs.append(sign_dict[ch])
                    elif ch in number_dict:
                        signs.append(number_dict[ch])
                    elif ch in punct_dict:
                        signs.append(punct_dict[ch])

                words_signs.append({
                    "type": "letters",
                    "letters": signs
                })
            else:
                words_signs.append({
                    "type": "invalid",
                    "word": w
                })
                invalid_words.append(w)

            i += 1

        session["words_signs"] = words_signs
        session["text"] = text
        session["invalid"] = invalid_words
        return redirect(url_for("text_to_sign"))

    return render_template(
        "text_to_sign.html",
        words_signs=session.get("words_signs", []),
        text=session.get("text", ""),
        invalid=session.get("invalid", [])
    )

# # Dictionary of valid English words
# valid_words = set(words.words())

# # Letters → sign images (A-Z)
# sign_dict = {chr(i): f"signs/{chr(i)}.png" for i in range(65, 91)}

# # Numbers → sign images (0-9)
# number_dict = {str(i): f"signs/{i}.png" for i in range(10)}

# # Punctuation → sign images
# punct_dict = {
#     ".": "signs/fullstop.png",
#     ",": "signs/comma.png",
#     "!": "signs/exclamation.png",
#     "?": "signs/question.png"
# }

# # Optional: create a "space" placeholder image for spacing, or just leave empty string
# space_sign = "signs/space.png"  # you can create a blank image called space.png

# @app.route("/text-to-sign", methods=["GET", "POST"])
# def text_to_sign():
#     if request.method == "POST":
#         text = request.form.get("text", "")

#         words_signs = []
#         invalid_words = []

#         i = 0
#         tokens = text.split()
#         while i < len(tokens):
#             w = tokens[i]

#             # Check if a phrase starts with a slash
#             if w.startswith("/"):
#                 # Collect all tokens until closing slash
#                 phrase_tokens = [w[1:]]  # remove starting slash
#                 i += 1
#                 while i < len(tokens):
#                     token = tokens[i]
#                     if token.endswith("/"):
#                         phrase_tokens.append(token[:-1])  # remove ending slash
#                         break
#                     else:
#                         phrase_tokens.append(token)
#                     i += 1

#                 phrase = ' '.join(phrase_tokens).upper()
#                 signs = []
#                 for ch in phrase:
#                     if ch == ' ':
#                         signs.append(space_sign)  # keep a visual space
#                     elif ch in sign_dict:
#                         signs.append(sign_dict[ch])
#                     elif ch in number_dict:
#                         signs.append(number_dict[ch])
#                     elif ch in punct_dict:
#                         signs.append(punct_dict[ch])
#                 if signs:
#                     words_signs.append(signs)

#             else:
#                 # Regular word handling
#                 word_clean = ''.join([ch for ch in w if ch.isalpha()])
#                 digits_only = ''.join([ch for ch in w if ch.isdigit()])
#                 punct_only = ''.join([ch for ch in w if ch in punct_dict])

#                 if (word_clean.lower() in valid_words) or digits_only or punct_only:
#                     signs = []
#                     for ch in w.upper():
#                         if ch in sign_dict:
#                             signs.append(sign_dict[ch])
#                         elif ch in number_dict:
#                             signs.append(number_dict[ch])
#                         elif ch in punct_dict:
#                             signs.append(punct_dict[ch])
#                     if signs:
#                         words_signs.append(signs)
#                 else:
#                     words_signs.append([f"Incorrect word: {w}"])
#                     invalid_words.append(w)
#             i += 1

#         session["words_signs"] = words_signs
#         session["text"] = text
#         session["invalid"] = invalid_words
#         return redirect(url_for("text_to_sign"))

#     words_signs = session.get("words_signs", [])
#     text = session.get("text", "")
#     invalid_words = session.get("invalid", [])
#     return render_template("text_to_sign.html", words_signs=words_signs, text=text, invalid=invalid_words)

  
        
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
