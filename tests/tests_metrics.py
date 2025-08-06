import pandas as pd
from metrics.evaluator import evaluate_model

def evaluate_all_model_outputs(csv_path: str):
    """
    Avalia múltiplos modelos (output1 e output2) comparando com y_true_1 e y_true_2.
    Chama evaluate_model() para cada par e imprime/retorna métricas.
    """

    df = pd.read_csv(csv_path)

    # Valores reais
    y_true_1 = df["y_true_1"].values
    y_true_2 = df["y_true_2"].values

    # Modelos
    model_outputs = {
        "Model 1 - Output 1": df["model1_out1"].values,
        "Model 1 - Output 2": df["model1_out2"].values,
        "Model 2 - Output 1": df["model2_out1"].values,
        "Model 2 - Output 2": df["model2_out2"].values,
    }

    y_trues = {
        "Output 1": y_true_1,
        "Output 2": y_true_2,
    }

    # Avaliação
    results = {}
    for name, y_pred in model_outputs.items():
        output_type = "Output 1" if "Output 1" in name else "Output 2"
        y_true = y_trues[output_type]

        print(f"\n🔍 Avaliando: {name}")
        metrics = evaluate_model(y_true, y_pred)
        results[name] = metrics

    return results

evaluate_all_model_outputs("data/sweep_output.csv")