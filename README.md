# ml-evaluator

📊 A web-based dashboard for evaluating and comparing machine learning model regression.

## Features
- Upload CSV files containing `y_true` and regression from one or more models
- View metrics: Accuracy, Precision, Recall, F1 Score
- Visualize Confusion Matrix and ROC Curve
- Compare different models from a dropdown menu

## Example
🚨 This example uses models trained by a Bayesian Algorithm (*Project*: [Black_Box](https://github.com/GalbierY/Black_Box)), this type of algorithm tends to be more assertive in Pareto objective areas, having many errors far from the objective.
Example CSV format:(*'scripts/create_csv_by_model.py'*)
```csv
input1,input2,y_true_1,y_true_2,model1_out1,model1_out2,model2_out1,model2_out2
```

## Usage
```bash
pip install -r requirements.txt
streamlit run app.py
```

Or using Docker:
```bash
docker build -t model-dashboard .
docker run -p 8501:8501 model-dashboard
```

## Run Tests
```bash
pytest
```
