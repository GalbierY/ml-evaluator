import streamlit as st
import pandas as pd
import numpy as np
from metrics.evaluator import evaluate_model, plot_confusion_matrix, plot_roc_curve

# streamlit run app.py

st.set_page_config(page_title="Model Metrics Dashboard", layout="wide")
st.title("📊 Model Performance Dashboard")

uploaded_file = st.file_uploader("Upload CSV with predictions and ground truth", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df)

    # Identificar colunas disponíveis
    columns = list(df.columns)

    # Separar colunas possíveis para avaliação
    y_true_cols = [col for col in columns if col.startswith("y_true")]
    model_cols = [col for col in columns if col.startswith("model")]

    if not y_true_cols or not model_cols:
        st.error("CSV must contain at least one 'y_true_' and one 'model_' column.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            selected_y_true = st.selectbox("Select Ground Truth Column", y_true_cols)
        with col2:
            selected_model = st.selectbox("Select Model Output Column", model_cols)

        y_true_raw = df[selected_y_true].values
        y_pred_raw = df[selected_model].values

        # Converter para binário automaticamente se forem contínuos
        threshold = st.slider("Threshold for classification", 0.0, 1.0, 0.5, step=0.01)
        y_true = (y_true_raw > threshold).astype(int)
        y_pred = (y_pred_raw > threshold).astype(int)

        st.write("### Evaluation Metrics")
        metrics = evaluate_model(y_true, y_pred)
        st.json(metrics)

        st.write("### Confusion Matrix")
        st.pyplot(plot_confusion_matrix(y_true, y_pred))

        st.write("### ROC Curve")
        st.pyplot(plot_roc_curve(y_true, y_pred))
