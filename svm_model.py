# 1️⃣ Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


df = pd.read_csv("data/diabetes.csv")

# 3️⃣ Inspect Dataset (optional but recommended)
print("First 5 rows:\n", df.head())
print("\nDataset Info:\n", df.info())
print("\nSummary Statistics:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# 4️⃣ Define Features and Labels
X = df.drop("Outcome", axis=1)  # Features
y = df["Outcome"]               # Target

# 5️⃣ Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6️⃣ Scale Features (SVM works better with scaled data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7️⃣ Train SVM Model
model = SVC(kernel='rbf', C=1.0, gamma='scale')  # RBF kernel
model.fit(X_train_scaled, y_train)

# 8️⃣ Make Predictions
y_pred = model.predict(X_test_scaled)

# 9️⃣ Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, "svm_diabetes_model.pkl", compress=3)
joblib.dump(scaler, "scaler.pkl", compress=3)

print("\nModel and scaler saved successfully!")
