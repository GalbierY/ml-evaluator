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


def sweep_and_generate_csv(model1_out1_path, model1_out2_path, model2_out1_path, model2_out2_path, output_csv='data/sweep_results.csv'):
    num_points_x1 = int(round((150 - 0) / 0.5)) + 1
    num_points_x2 = int(round((1 - 0) / 0.01)) + 1
    x1_values = np.linspace(0, 150, num_points_x1)
    x2_values = np.linspace(0, 1, num_points_x2)
    grid = [(x1, x2) for x1 in x1_values for x2 in x2_values]

    model1_out1 = load_model(model1_out1_path)
    model1_out2 = load_model(model1_out2_path)

    model2_out1 = load_model(model2_out1_path)
    model2_out2 = load_model(model2_out2_path)

    results = []

    for x1, x2 in grid:
        y_true_1, y_true_2 = dummy_function.Simulation_Env_maneuver.black_box_energy_management(x1, x2)
        X_input = np.array([[x1, x2]])

        m1_out1 = model1_out1.predict(X_input)[0]
        m1_out2 = model1_out2.predict(X_input)[0]

        m2_out1 = model2_out1.predict(X_input)[0]
        m2_out2 = model2_out2.predict(X_input)[0]

        results.append({
            'input1': x1,
            'input2': x2,
            'y_true_1': y_true_1,
            'y_true_2': y_true_2,
            'model1_out1': m1_out1,
            'model1_out2': m1_out2,
            'model2_out1': m2_out1,
            'model2_out2': m2_out2,
        })
        
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f'✅ CSV salvo: {output_csv}')

sweep_and_generate_csv(
    'models/Krigin_model_out1_100.pkl',
    'models/Krigin_model_out2_100.pkl',
    'models/Krigin_model_out1_300.pkl',
    'models/Krigin_model_out2_300.pkl',
    output_csv='data/sweep_output.csv'
)