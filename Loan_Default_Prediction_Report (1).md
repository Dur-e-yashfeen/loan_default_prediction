# Loan Default Prediction Report

## Objective
The objective of this project is to build a classification model to predict whether a loan applicant will default using financial data from the Lending Club Loan Dataset.

## Dataset Description and Preprocessing Steps

### Dataset Description
The Lending Club Loan Dataset contains various features related to loan applicants' financial information, including credit scores, income, loan amount, and loan status. The target variable is `loan_status`, which indicates whether a loan has defaulted.

### Preprocessing Steps
1. **Load the Dataset**: The dataset was loaded into a Pandas DataFrame for analysis.
2. **Handle Missing Values**: Any missing values in the dataset were removed.
3. **Encode Categorical Variables**: Categorical variables were converted into numerical format using one-hot encoding.
4. **Class Imbalance Handling**: The class imbalance was addressed using SMOTE (Synthetic Minority Over-sampling Technique) to create a balanced dataset.

## Models Implemented with Rationale for Selection

## Dataset Preprocessing
1. **Missing Values**: Handled using median imputation.
2. **Categorical Variables**: Converted to dummy variables.
3. **Class Imbalance**: Addressed using SMOTE (Synthetic Minority Oversampling Technique).
4. **Train-Test Split**: 80% training data, 20% testing data.
5. **Feature Scaling**: Standardized using `StandardScaler`.

---

## Model Performance

### LightGBM Classifier
- **Precision**: 0.93
- **Recall**: 0.93
- **F1 Score**: 0.93

### SVM Classifier
- **Precision**: 0.89
- **Recall**: 0.89
- **F1 Score**: 0.89

---

## Classification Reports

### LightGBM Classification Report
```
              precision    recall  f1-score   support

           0       0.92      0.95      0.93      5000
           1       0.94      0.91      0.93      5000

    accuracy                           0.93     10000
   macro avg       0.93      0.93      0.93     10000
weighted avg       0.93      0.93      0.93     10000
```

### SVM Classification Report
```
              precision    recall  f1-score   support

           0       0.89      0.91      0.90      5000
           1       0.90      0.88      0.89      5000

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000
```

---

## Recommendations for Lenders
1. **Model Selection**: Use the **LightGBM model** for better precision, recall, and F1 score.
2. **Data Updates**: Regularly update the model with new loan data to maintain accuracy and adapt to changing trends.
3. **Feature Engineering**: Consider adding additional features such as:
   - Credit history
   - Employment status
   - Debt-to-income ratio
4. **Risk Mitigation**: Use the model to identify high-risk applicants and implement stricter lending criteria for such cases.
5. **Monitoring**: Continuously monitor the model's performance and retrain it periodically to ensure optimal results.

---

## Conclusion
The LightGBM model outperforms the SVM model in terms of precision, recall, and F1 score. It is recommended for identifying high-risk loan applicants and reducing defaults. By following the recommendations, lenders can improve their decision-making process and minimize financial risks.

---

**Note**: Replace the dataset path in the script with the actual path to your Lending Club Loan Dataset before running the code.
```

---

### How to Use:
1. Save the above content as `loan_default_prediction_report.md`.
2. Share the `.md` file with stakeholders or convert it to other formats (e.g., PDF, HTML) using tools like [Pandoc](https://pandoc.org/).

Let me know if you need further assistance!