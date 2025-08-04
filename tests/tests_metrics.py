import numpy as np
from metrics.evaluator import evaluate_model

def test_evaluate_model():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = evaluate_model(y_true, y_pred)
    assert isinstance(metrics, dict)
    assert "Accuracy" in metrics
    assert "F1 Score" in metrics
    assert metrics["Accuracy"] >= 0 and metrics["Accuracy"] <= 1