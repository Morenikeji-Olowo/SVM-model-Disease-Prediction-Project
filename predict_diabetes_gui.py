import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("svm_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function to make prediction
def predict():
    try:
        # Get input values
        data = [
            float(entry_pregnancies.get()),
            float(entry_glucose.get()),
            float(entry_bp.get()),
            float(entry_skin.get()),
            float(entry_insulin.get()),
            float(entry_bmi.get()),
            float(entry_dpf.get()),
            float(entry_age.get())
        ]
        
        data_array = np.array([data])
        data_scaled = scaler.transform(data_array)
        prediction = model.predict(data_scaled)
        
        if prediction[0] == 1:
            result.set("Likely to have diabetes")
        else:
            result.set("Unlikely to have diabetes")
    
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")

# Create main window
root = tk.Tk()
root.title("Diabetes Prediction")

# Input fields
labels = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
          "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

entries = []
for i, text in enumerate(labels):
    tk.Label(root, text=text).grid(row=i, column=0, padx=10, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

entry_pregnancies, entry_glucose, entry_bp, entry_skin, entry_insulin, entry_bmi, entry_dpf, entry_age = entries

# Predict button
tk.Button(root, text="Predict", command=predict).grid(row=8, column=0, columnspan=2, pady=10)

# Result display
result = tk.StringVar()
tk.Label(root, textvariable=result, font=("Arial", 14)).grid(row=9, column=0, columnspan=2, pady=10)

# Start GUI loop
root.mainloop()

