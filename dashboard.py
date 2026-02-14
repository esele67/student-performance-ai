# dashboard.py
import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
from io import BytesIO
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Academic Intelligence System", layout="wide")

# -----------------------------
# 🔥 PREMIUM HEADER DESIGN
# -----------------------------
st.markdown("""
<style>
.big-title {
    font-size:45px !important;
    font-weight:700;
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    font-size:18px;
    color:gray;
}
.metric-card {
    background: #0e1117;
    padding: 20px;
    border-radius: 15px;
    text-align:center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🎓 AI Academic Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Next-generation student performance analytics & prediction platform</div>', unsafe_allow_html=True)
st.write("")

# -----------------------------
# SAFE MODEL LOADING (Cloud Safe)
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/rf_model_compat.joblib")
        features = joblib.load("models/training_features.joblib")
        return model, features
    except Exception:
        st.error("❌ Model files missing. Check GitHub models/ folder.")
        st.stop()

rf_model, training_features = load_model()

# -----------------------------
# FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload Student CSV File", type="csv")

# -----------------------------
# IF FILE UPLOADED
# -----------------------------
if uploaded_file:

    # 🔥 AI LOADING EFFECT
    with st.spinner("🧠 AI analyzing academic patterns..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)

    st.success("AI analysis complete!")

    # -----------------------------
    # READ DATA
    # -----------------------------
    data = pd.read_csv(uploaded_file, sep=';')

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(data.head())

    # Remove real result if exists
    if "G3" in data.columns:
        data = data.drop("G3", axis=1)

    # Encode categorical
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Match training features
    missing_cols = set(training_features) - set(data.columns)
    for col in missing_cols:
        data[col] = 0

    data = data[training_features]

    # -----------------------------
    # AI PREDICTIONS
    # -----------------------------
    predictions = rf_model.predict(data)
    data["Predicted Grade"] = predictions.round(1)

    def risk_label(x):
        if x < 10:
            return "At Risk"
        elif x < 14:
            return "Average"
        else:
            return "Excellent"

    data["Risk Level"] = data["Predicted Grade"].apply(risk_label)

    # -----------------------------
    # AI COMMENTS
    # -----------------------------
    def explain(row):
        g = row["Predicted Grade"]
        if g < 10:
            return f"High failure risk. Score {g}/20. Needs urgent academic support."
        elif g < 14:
            return f"Average performance ({g}/20). Can improve with guidance."
        else:
            return f"Excellent performance ({g}/20). Likely top performer."

    data["AI Comment"] = data.apply(explain, axis=1)

    # -----------------------------
    # EXECUTIVE DASHBOARD
    # -----------------------------
    st.divider()
    st.subheader("📊 Executive AI Dashboard")

    avg_grade = data["Predicted Grade"].mean()
    total_students = len(data)
    risk_count = (data["Risk Level"] == "At Risk").sum()
    excellent_count = (data["Risk Level"] == "Excellent").sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("👨‍🎓 Students", total_students)
    c2.metric("⚠️ At Risk", risk_count)
    c3.metric("🏆 Excellent", excellent_count)
    c4.metric("📈 Class Avg", f"{avg_grade:.2f}/20")

    st.divider()

    # -----------------------------
    # TOP STUDENTS
    # -----------------------------
    st.subheader("🏆 Top Performing Students")
    top_students = data.sort_values(by="Predicted Grade", ascending=False).head(5)
    st.dataframe(top_students[["Predicted Grade", "Risk Level"]])

    # -----------------------------
    # AT RISK STUDENTS
    # -----------------------------
    st.subheader("⚠️ Students Needing Attention")
    risk_students = data[data["Risk Level"] == "At Risk"]
    st.dataframe(risk_students[["Predicted Grade", "AI Comment"]])

    # -----------------------------
    # FULL TABLE
    # -----------------------------
    st.subheader("🧠 Full AI Predictions")
    st.dataframe(data)

    # -----------------------------
    # AI INSIGHTS
    # -----------------------------
    st.subheader("🤖 AI Insights")

    if avg_grade < 10:
        st.error("Overall class performance is poor. Immediate intervention required.")
    elif avg_grade < 14:
        st.warning("Class performance is average. Improvement plan recommended.")
    else:
        st.success("Class performing excellently overall.")

    if risk_count > total_students * 0.4:
        st.error("High number of failing students detected!")

    st.divider()

    # -----------------------------
    # DOWNLOAD UPDATED CSV
    # -----------------------------
    csv = data.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇️ Download Updated CSV with Predictions",
        data=csv,
        file_name="AI_updated_students.csv",
        mime="text/csv"
    )

    # -----------------------------
    # PDF REPORT
    # -----------------------------
    if st.button("📄 Generate Full AI Report"):

        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 12, "AI Academic Intelligence Report", ln=True, align="C")
        pdf.ln(8)

        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Class Summary", ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Total Students: {total_students}", ln=True)
        pdf.cell(0, 8, f"Class Average: {avg_grade:.2f}/20", ln=True)
        pdf.cell(0, 8, f"At Risk Students: {risk_count}", ln=True)
        pdf.cell(0, 8, f"Excellent Students: {excellent_count}", ln=True)

        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Student Predictions", ln=True)
        pdf.set_font("Arial", "", 11)

        for i, row in data.iterrows():
            comment = str(row["AI Comment"]).encode("latin-1", "ignore").decode("latin-1")

            pdf.multi_cell(
                0, 7,
                f"Student {i+1}\n"
                f"Predicted Grade: {row['Predicted Grade']}/20\n"
                f"Risk Level: {row['Risk Level']}\n"
                f"{comment}"
            )
            pdf.ln(2)

        pdf_output = pdf.output(dest="S").encode("latin-1")
        pdf_buffer = BytesIO(pdf_output)

        st.success("Report generated successfully!")

        st.download_button(
            label="⬇️ Download AI PDF Report",
            data=pdf_buffer,
            file_name="AI_student_report.pdf",
            mime="application/pdf"
        )

else:
    st.info("👆 Upload a student CSV file to activate the AI system.")
