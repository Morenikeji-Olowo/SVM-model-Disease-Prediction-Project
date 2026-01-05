import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("svm_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

print("=== Diabetes Prediction Program ===")

while True:
    try:
        print("\nEnter new patient data (or type 'exit' to quit):")

        pregnancies = input("Pregnancies: ")
        if pregnancies.lower() == "exit":
            break
        pregnancies = int(pregnancies)

        glucose = input("Glucose: ")
        if glucose.lower() == "exit":
            break
        glucose = float(glucose)

        blood_pressure = input("BloodPressure: ")
        if blood_pressure.lower() == "exit":
            break
        blood_pressure = float(blood_pressure)

        skin_thickness = input("SkinThickness: ")
        if skin_thickness.lower() == "exit":
            break
        skin_thickness = float(skin_thickness)

        insulin = input("Insulin: ")
        if insulin.lower() == "exit":
            break
        insulin = float(insulin)

        bmi = input("BMI: ")
        if bmi.lower() == "exit":
            break
        bmi = float(bmi)

        dpf = input("DiabetesPedigreeFunction: ")
        if dpf.lower() == "exit":
            break
        dpf = float(dpf)

        age = input("Age: ")
        if age.lower() == "exit":
            break
        age = int(age)

        # Prepare input array and scale
        new_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                              insulin, bmi, dpf, age]])
        new_data_scaled = scaler.transform(new_data)

        # Make prediction
        prediction = model.predict(new_data_scaled)

        # Display result
        if prediction[0] == 1:
            print("Prediction: The patient is likely to have diabetes.")
        else:
            print("Prediction: The patient is unlikely to have diabetes.")

    except ValueError:
        print("Invalid input. Please enter numeric values for all features.")

print("\nExiting program. Goodbye!")
