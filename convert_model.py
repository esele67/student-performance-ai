import joblib

# Load your existing model
rf_model = joblib.load("models/rf_model.joblib")

# Re-save using protocol=4 (compatible with most Python versions)
joblib.dump(rf_model, "models/rf_model_compat.joblib", protocol=4)

print("✅ Model re-saved successfully as rf_model_compat.joblib")
