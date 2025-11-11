import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# --------------------------
# Page Setup
# --------------------------
st.set_page_config(page_title="AI Diabetes Predictor Dashboard", layout="wide")
# Custom CSS for tab styling
st.markdown(
    """
    <style>
    /* Change inactive tab color */
    .stTabs [data-baseweb="tab"] {
        color: #1E90FF; /* Text color for inactive tabs (dodger blue) */
        background-color: #F0F8FF; /* Light background (AliceBlue) */
        border-radius: 10px;
        padding: 8px 16px;
        margin-right: 5px;
    }

    /* Change active tab color */
    .stTabs [aria-selected="true"] {
        color: white !important;
        background-color: #0A3D62 !important; /* Dark royal blue */
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------
# Load Data
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

@st.cache_data
def load_model_results():
    return pd.read_csv("model_results.csv")

@st.cache_resource
def load_best_model():
    return joblib.load("best_model.pkl")

df = load_data()
results = load_model_results()
best_model = load_best_model()

# --------------------------
# Tabs
# --------------------------
tabs = st.tabs(["📊 Data Overview", "🤖 Model Comparison", "🔮 Live Prediction"])

# ==========================
# TAB 1 — DATA OVERVIEW
# ==========================
with tabs[0]:
    st.header("📊 Data Overview & Insights")
    st.write("This section provides an overview of the diabetes dataset and visual insights into key health parameters.")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Basic Statistics")
    st.write(df.describe())

    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(8,5))
    sns.heatmap(df.corr(), annot=True, cmap="Blues")
    st.pyplot(plt)

    st.subheader("Distribution of Target Variable")
    fig, ax = plt.subplots()
    sns.countplot(x='Outcome', data=df, palette='Blues', ax=ax)
    ax.set_xticklabels(['Non-Diabetic', 'Diabetic'])
    st.pyplot(fig)

# ==========================
# TAB 2 — MODEL COMPARISON
# ==========================
with tabs[1]:
    st.header("🤖 Model Performance Comparison")
    st.write("This section compares accuracy and other metrics for all trained models.")

    st.subheader("Model Scores Table")
    st.dataframe(results)

    st.subheader("Model Accuracy Comparison")
    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="Accuracy", data=results, palette="Blues_d", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    best_model_name = results.loc[results['Accuracy'].idxmax(), 'Model']
    st.success(f"🏆 **Best Performing Model:** {best_model_name}")

# ==========================
# TAB 3 — LIVE PREDICTION
# ==========================
with tabs[2]:
    st.header("🔮 Live Prediction")
    st.write("Enter health metrics below to predict if a person is likely to have diabetes.")

    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
        Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
        SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    with col2:
        Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        Age = st.number_input("Age", min_value=1, max_value=120, value=30)

    if st.button("Get Prediction"):
        user_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigreeFunction, Age]]
        prediction = best_model.predict(user_data)[0]
        if prediction == 1:
            st.error("🔴 The person is **likely Diabetic**.")
        else:
            st.success("🟢 The person is **likely Non-Diabetic**.")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("Developed with 💙 using Streamlit | AI Diabetes Predictor © 2025")
