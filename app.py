import streamlit as st
import pandas as pd
from metrics.evaluator import evaluate_model, plot_confusion_matrix, plot_roc_curve

st.set_page_config(page_title="Model Metrics Dashboard", layout="wide")
st.title("📊 Model Performance Dashboard")

uploaded_file = st.file_uploader("Upload CSV with y_true and model predictions", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    columns = list(df.columns)
    if "y_true" not in columns:
        st.error("CSV must contain a 'y_true' column.")
    else:
        model_columns = [col for col in columns if col != "y_true"]

        selected_model = st.selectbox("Select Model to Evaluate", model_columns)

        y_true = df["y_true"].values
        y_pred = df[selected_model].values

        metrics = evaluate_model(y_true, y_pred)
        st.write("### Evaluation Metrics")
        st.json(metrics)

        st.write("### Confusion Matrix")
        st.pyplot(plot_confusion_matrix(y_true, y_pred))

        st.write("### ROC Curve")
        st.pyplot(plot_roc_curve(y_true, y_pred))
