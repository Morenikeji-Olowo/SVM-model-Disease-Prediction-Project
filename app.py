from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)

# ENABLE CORS
CORS(app)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://svm-phi.vercel.app"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

model = joblib.load("svm_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        features = np.array([[
            data["pregnancies"],
            data["glucose"],
            data["blood_pressure"],
            data["skin_thickness"],
            data["insulin"],
            data["bmi"],
            data["dpf"],
            data["age"]
        ]])

        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        return jsonify({
            "success": True,
            "prediction": int(prediction),
            "result": "Likely diabetic" if prediction == 1 else "Unlikely diabetic"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
