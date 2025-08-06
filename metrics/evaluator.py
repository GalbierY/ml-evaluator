from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2_score(y_true, y_pred)
    }

def plot_true_vs_pred(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # linha y=x
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("True vs Predicted")
    return fig

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_xlabel("Residuals (True - Predicted)")
    ax.set_title("Residuals Distribution")
    return fig

def plot_residuals_vs_pred(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, linestyle='--', color='red')
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted Values")
    return fig

def plot_prediction_distribution(y_true, y_pred):
    fig, ax = plt.subplots()
    sns.kdeplot(y_true, label="True", ax=ax)
    sns.kdeplot(y_pred, label="Predicted", ax=ax)
    ax.set_title("Distribution: True vs Predicted")
    ax.legend()
    return fig

def plot_binned_errors(y_true, y_pred, bins=10):
    df = pd.DataFrame({'y_true': y_true, 'error': y_true - y_pred})
    df['bin'] = pd.cut(df['y_true'], bins=bins)
    error_by_bin = df.groupby('bin')['error'].mean()

    fig, ax = plt.subplots()
    error_by_bin.plot(kind='bar', ax=ax)
    ax.set_ylabel("Mean Error")
    ax.set_title("Mean Error by True Value Bin")
    return fig
