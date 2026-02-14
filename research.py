# research.py

# -----------------------------
# Step 1: Import libraries
# -----------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import shap
import matplotlib.pyplot as plt
import os

# -----------------------------
# Step 2: Load dataset
# -----------------------------
data = pd.read_csv("student-mat.csv", sep=';')

# Preview dataset
print(data.head())
print(data.info())

# -----------------------------
# Step 3: Preprocess categorical columns
# -----------------------------
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# -----------------------------
# Step 4: Separate features and target
# -----------------------------
X = data.drop("G3", axis=1)
y = data["G3"]

# -----------------------------
# Step 5: Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 6: Train Random Forest
# -----------------------------
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42
)
rf_model.fit(X_train, y_train)

# -----------------------------
# Step 7: Predict and evaluate
# -----------------------------
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest MSE:", mse)
print("Random Forest R^2:", r2)

# -----------------------------
# Step 8: Save the trained model and feature names
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/rf_model.joblib")
joblib.dump(X_train.columns.tolist(), "models/training_features.joblib")
print("Model saved in models/rf_model.joblib")
print("Training features saved in models/training_features.joblib")

# -----------------------------
# Step 9: Explain predictions with SHAP
# -----------------------------
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Summary plot of top features
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("models/feature_importance.png")
print("Feature importance plot saved as models/feature_importance.png")
