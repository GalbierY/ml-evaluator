import numpy as np
from metrics.evaluator import evaluate_model

def test_evaluate_model():
    # Simular sweeping com x1 de 0 a 150 e x2 de 0 a 1
    x1 = np.linspace(0, 150, 100)
    x2 = np.linspace(0, 1, 100)
    y_true = (x1 + x2 > 75).astype(int)  # Exemplo de classificação binária
    y_pred = (x1 + x2 > 80).astype(int)  # Leve deslocamento proposital

    metrics = evaluate_model(y_true, y_pred)

    assert isinstance(metrics, dict)
    assert "Accuracy" in metrics
    assert "F1 Score" in metrics
    assert 0.0 <= metrics["Accuracy"] <= 1.0