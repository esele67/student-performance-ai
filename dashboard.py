import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from fpdf import FPDF

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance AI Dashboard")
st.markdown("Upload student data to predict academic performance risk.")

# ---------------------------------------------------
# Safe Model Loading (Cloud-Ready)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/rf_model_compat.joblib")
        features = joblib.load("models/training_features.joblib")
        return model, features
    except Exception as e:
        st.error("❌ Model files missing or corrupted.")
        st.error("Make sure the 'models/' folder is uploaded to GitHub.")
        st.stop()

rf_model, training_features = load_model()

# ---------------------------------------------------
# File Upload
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Student CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)  # auto-detect separator
    except Exception:
        st.error("❌ Could not read CSV file.")
        st.stop()

    st.success("✅ File uploaded successfully!")
    st.write("Preview of Data:")
    st.dataframe(data.head())

    # ---------------------------------------------------
    # Prediction Section
    # ---------------------------------------------------
    if st.button("🚀 Run AI Prediction"):

        with st.spinner("🧠 AI analyzing student performance..."):
            time.sleep(1)

        try:
            # Ensure columns match training features
            missing_cols = set(training_features) - set(data.columns)
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.stop()

            X = data[training_features]

            predictions = rf_model.predict(X)

            data["Prediction"] = predictions

            st.success("✅ Prediction Completed!")

            # Display results
            st.subheader("📊 Prediction Results")
            st.dataframe(data)

            # ---------------------------------------------------
            # Download Results
            # ---------------------------------------------------
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Results as CSV",
                csv,
                "predictions.csv",
                "text/csv"
            )

            # ---------------------------------------------------
            # Generate PDF Report
            # ---------------------------------------------------
            if st.button("📄 Generate PDF Report"):

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=10)

                pdf.cell(200, 10, txt="Student Performance AI Report", ln=True, align="C")
                pdf.ln(10)

                for col in data.columns:
                    pdf.cell(40, 8, col, border=1)
                pdf.ln()

                for _, row in data.head(20).iterrows():
                    for item in row:
                        pdf.cell(40, 8, str(item), border=1)
                    pdf.ln()

                pdf_output = "report.pdf"
                pdf.output(pdf_output)

                with open(pdf_output, "rb") as f:
                    st.download_button(
                        "📥 Download PDF Report",
                        f,
                        file_name="student_report.pdf",
                        mime="application/pdf"
                    )

        except Exception as e:
            st.error("❌ Prediction failed.")
            st.write(str(e))
