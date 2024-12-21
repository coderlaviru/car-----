# Car Price Prediction Project

## Overview
This project involves predicting car prices based on various features using machine learning models. The pipeline includes data preprocessing, feature engineering, model training, evaluation, and comparative analysis of multiple models.

---

## Data Preparation

### Data Preprocessing
- Data is read using `pandas` and checked for null values and inconsistencies.
- Continuous and categorical features are separated and treated accordingly.
- Data normalization and encoding are applied as required.

### Feature Engineering
- Domain knowledge is used to create new features or transform existing ones.
- Feature selection is performed to reduce dimensionality and improve model efficiency.

---

## Models Used
The following machine learning models were implemented and evaluated:

1. **Support Vector Regressor (SVR)**
2. **Multilayer Perceptron Regressor (MLP)**
3. **Naive Bayes (Gaussian)**
4. **XGBoost Regressor**

---

## Evaluation Metrics

### Regression Metrics
- **Mean Absolute Error (MAE)**: Measures average magnitude of errors in predictions.
- **Mean Squared Error (MSE)**: Penalizes larger errors more heavily.
- **Root Mean Squared Error (RMSE)**: Square root of MSE for interpretable units.
- **R-squared (R²)**: Indicates the proportion of variance explained by the model.

### Classification Metrics (if applicable):
- **Accuracy**: Percentage of correct predictions.
- **Precision**: Proportion of true positives among predicted positives.
- **Recall**: Proportion of true positives among actual positives.
- **F1-Score**: Harmonic mean of precision and recall.

---

## Analysis and Results

### Model Performance
- **XGBoost Regressor** achieved the lowest RMSE and highest R², making it the best-performing model.
- **Multilayer Perceptron** showed promising results but required more tuning.
- **SVR** struggled with complex relationships in the dataset.

### Error Handling
- Classification metrics were initially calculated, but regression tasks require adapting these metrics to continuous targets.
- Adjustments were made to use regression-specific metrics.

---

## Visualizations

### Key Insights:
- Correlation heatmaps to identify feature relationships.
- Actual vs. Predicted plots for model evaluation.
- Residual plots to analyze error distribution.

---

## Steps to Reproduce

### Prerequisites
- Python 3.12
- Anaconda environment (or preferred virtual environment)

# Model Evaluation Metrics

This project evaluates the performance of multiple machine learning models, including SVC, MLP, Naive Bayes, and XGBoost. The evaluation metrics used depend on whether the problem is a classification or regression task.

---

## Project Workflow

### 1. **Define Models and Predictions**
We store model predictions in a dictionary for easier access and evaluation:


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




### 4. **Display Results**
The results are stored in a Pandas DataFrame for clear visualization:


## Error Handling

### Common Issues:
1. **Continuous Target in Classification**:
   - Ensure the target variable (`y_test`) contains discrete class labels.
   - If the task is regression, switch to appropriate regression metrics.

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
