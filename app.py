from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)

# Enable CORS for your Vercel frontend only
CORS(app, origins=["https://svm-phi.vercel.app"])

# Load model and scaler
model = joblib.load("svm_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Preflight requests for CORS
    if request.method == "OPTIONS":
        return "", 200

    try:
        data = request.json
        print("Received data:", data)  # Debug log

        # Safely extract all features and convert to float
        pregnancies = float(data.get("pregnancies", 0))
        glucose = float(data.get("glucose", 0))
        blood_pressure = float(data.get("blood_pressure", 0))
        skin_thickness = float(data.get("skin_thickness", 0))
        insulin = float(data.get("insulin", 0))
        bmi = float(data.get("bmi", 0))
        # Accept either dpf or diabetes_pedigree_function
        dpf = float(data.get("dpf", data.get("diabetes_pedigree_function", 0)))
        age = float(data.get("age", 0))

        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                              insulin, bmi, dpf, age]])

        # Scale and predict
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        return jsonify({
            "success": True,
            "prediction": int(prediction),
            "result": "Likely diabetic" if prediction == 1 else "Unlikely diabetic"
        })

    except Exception as e:
        print("Error in /predict:", e)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
