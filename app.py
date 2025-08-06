import streamlit as st
import pandas as pd
import numpy as np
from metrics.evaluator import evaluate_model, plot_true_vs_pred, plot_residuals, plot_residuals_vs_pred, plot_prediction_distribution, plot_binned_errors

# streamlit run app.py

st.set_page_config(page_title="Model Metrics Dashboard", layout="wide")
st.title("📊 Model Performance Dashboard")

uploaded_file = st.file_uploader("Upload CSV with predictions and ground truth", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df)

    columns = list(df.columns)
    y_true_cols = [col for col in columns if "y_true" in col.lower()]
    model_cols = [col for col in columns if "model" in col.lower()]

    if not y_true_cols or not model_cols:
        st.error("CSV must contain at least one 'y_true_' and one 'model_' column.")
    else:
        st.markdown("### 🧠 Select Multiple Pairs to Compare")

        pairs = []
        for true_col in y_true_cols:
            related_preds = [col for col in model_cols if true_col.split("_")[-1] in col]
            for pred_col in related_preds:
                pairs.append((true_col, pred_col))

        selected_pairs = st.multiselect("Select (Ground Truth, Prediction) Pairs", pairs, default=pairs)

        for idx, (true_col, pred_col) in enumerate(selected_pairs):
            y_true = df[true_col].values
            y_pred = df[pred_col].values

            st.markdown(f"---\n### 🔎 Evaluation for `{true_col}` vs `{pred_col}`")

            metrics = evaluate_model(y_true, y_pred)
            st.json(metrics)

            cm_col, roc_col, res_pred_col, pred_dist_col, bin_er_col = st.columns(5)

            with cm_col:
                st.pyplot(plot_true_vs_pred(y_true, y_pred))

            with roc_col:
                st.pyplot(plot_residuals(y_true, y_pred))
            
            with res_pred_col:
                st.pyplot(plot_residuals_vs_pred(y_true, y_pred))

            with pred_dist_col:
                st.pyplot(plot_prediction_distribution(y_true, y_pred))

            with bin_er_col:
                st.pyplot(plot_binned_errors(y_true, y_pred))
