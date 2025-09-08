from flask import Flask, request, jsonify
import pickle
import numpy as np
import cv2
import os
import mediapipe as mp
from flask_cors import CORS  
app = Flask(__name__)
CORS(app)  
# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Mediapipe setup
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
    # mp_hands = mp.solutions.hands
    # hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)
    # # mp_draw = mp.solutions.drawing_utils

    results = hands.process(rgb)
    
    print("Hand detection:", results.multi_hand_landmarks) 
    if results.multi_hand_landmarks:
        # First detected hand
        handLms = results.multi_hand_landmarks[0]
        row = []
        for lm in handLms.landmark:
            row.extend([lm.x, lm.y, lm.z])

        features = np.array(row).reshape(1, -1)  # shape = (1, 63)

        prediction = model.predict(features)
        result = str(prediction[0])
        return jsonify({"prediction": result})
    else:
        return jsonify({"error": "No hand detected"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render देगा अपना port
    app.run(host="0.0.0.0", port=port,debug=False)
