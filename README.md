# ml-evaluator

📊 A web-based dashboard for evaluating and comparing machine learning model predictions.

## Features
- Upload CSV files containing `y_true` and predictions from one or more models
- View metrics: Accuracy, Precision, Recall, F1 Score
- Visualize Confusion Matrix and ROC Curve
- Compare different models from a dropdown menu

## Example
Example CSV format:
```csv
y_true,model1,model2
0,0,0
1,1,1
0,1,0
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
