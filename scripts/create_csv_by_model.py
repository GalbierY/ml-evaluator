import joblib
import pandas as pd
import numpy as np
import os
import dummy_function

def load_model(file_name):
    data = joblib.load(file_name)
    model = data['model']
    
    print('Model loaded successfully!')
    return model


def sweep_and_generate_csv(model1_path, model2_path, steps_x1=50, steps_x2=50, output_csv='data/sweep_results.csv'):
    x1_values = np.linspace(0, 150, steps_x1)
    x2_values = np.linspace(0, 1, steps_x2)
    grid = [(x1, x2) for x1 in x1_values for x2 in x2_values]

    model1 = load_model(model1_path)
    model2 = load_model(model2_path)

    results = []

    for x1, x2 in grid:
        y_true_1, y_true_2 = dummy_function.Simulation_Env_maneuver.black_box_energy_management(x1, x2)
        X_input = np.array([[x1, x2]])

        m1_out = model1.predict(X_input)[0]
        m2_out = model2.predict(X_input)[0]

        results.append({
            'input1': x1,
            'input2': x2,
            'y_true_1': y_true_1,
            'y_true_2': y_true_2,
            'model1_out': m1_out,
            'model2_out': m2_out,
        })
        
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f'✅ CSV salvo: {output_csv}')

sweep_and_generate_csv(
    model1_path='models/Krigin_model_out1.pkl',
    model2_path='models/Krigin_model_out2.pkl',
    steps_x1=50,
    steps_x2=50,
    output_csv='data/sweep_output.csv'
)