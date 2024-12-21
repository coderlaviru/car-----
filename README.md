# Model Evaluation Metrics

This project evaluates the performance of multiple machine learning models, including SVC, MLP, Naive Bayes, and XGBoost. The evaluation metrics used depend on whether the problem is a classification or regression task.

---

## Project Workflow

### 1. **Define Models and Predictions**
We store model predictions in a dictionary for easier access and evaluation:

```python
models = {
    'SVC': y_test_pred_svc,  # SVC model predictions
    'MLP': y_test_pred_mlp,  # MLP model predictions
    'Naive Bayes': y_test_pred_nb,  # Naive Bayes model predictions
    'XGBoost': y_test_pred_xg  # XGBoost model predictions
}
```

### 2. **Choose Metrics**
#### For Classification:
- **Accuracy**: Measures how often predictions match actual labels.
- **Precision**: Measures how many positive predictions are actually correct.
- **Recall**: Measures how many actual positives are correctly identified.
- **F1-Score**: Harmonic mean of precision and recall.

#### For Regression:
- **Mean Squared Error (MSE)**: Measures the average squared difference between predictions and actual values.
- **R-squared (R²)**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

### 3. **Evaluate Metrics**
For each model, the metrics are calculated and stored in a results table for easy comparison.

#### Classification Example:
```python
for model_name, y_pred in models.items():
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1_score = report['macro avg']['f1-score']

    results.append({
        'Model': model_name,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    })
```

#### Regression Example:
```python
for model_name, y_pred in models.items():
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': model_name,
        'Mean Squared Error': mse,
        'R-squared': r2
    })
```

### 4. **Display Results**
The results are stored in a Pandas DataFrame for clear visualization:

```python
results_df = pd.DataFrame(results)
print(results_df)
```

---

## Error Handling

### Common Issues:
1. **Continuous Target in Classification**:
   - Ensure the target variable (`y_test`) contains discrete class labels.
   - If the task is regression, switch to appropriate regression metrics.

   Example:
   ```python
   ValueError: continuous is not supported
   ```

2. **Mismatched Metrics**:
   - Use classification metrics for classification tasks and regression metrics for regression tasks.

---

## Example Results

### Classification Metrics Table:
| Model        | Accuracy | Precision | Recall | F1-Score |
|--------------|----------|-----------|--------|----------|
| SVC          | 0.92     | 0.91      | 0.90   | 0.90     |
| MLP          | 0.88     | 0.87      | 0.86   | 0.86     |
| Naive Bayes  | 0.85     | 0.84      | 0.83   | 0.83     |
| XGBoost      | 0.95     | 0.94      | 0.93   | 0.94     |

### Regression Metrics Table:
| Model        | Mean Squared Error | R-squared |
|--------------|--------------------|-----------|
| SVC          | 0.032              | 0.87      |
| MLP          | 0.045              | 0.82      |
| Naive Bayes  | 0.056              | 0.78      |
| XGBoost      | 0.021              | 0.91      |

---

## Notes
- This project assumes that predictions for all models (`y_test_pred_*`) and the actual test values (`y_test`) are available and correctly preprocessed.
- Ensure that your task (classification or regression) aligns with the chosen metrics and models.

---

## Dependencies
- Python 3.x
- scikit-learn
- pandas
- numpy
