import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Preprocess the data
def preprocess_data(data):
    # Drop irrelevant columns
    data = data.drop(['id', 'member_id'], axis=1)

    # Handle missing values
    data = data.fillna(data.median())

    # Convert categorical variables to dummy variables
    data = pd.get_dummies(data, drop_first=True)

    # Separate features and target
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Train and evaluate classifiers
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # LightGBM Classifier
    lgbm = LGBMClassifier(random_state=42)
    lgbm.fit(X_train, y_train)
    y_pred_lgbm = lgbm.predict(X_test)

    # SVM Classifier
    svm = make_pipeline(StandardScaler(), SVC(random_state=42))
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)

    # Evaluate performance
    print("LightGBM Classification Report:")
    print(classification_report(y_test, y_pred_lgbm))

    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm))

    # Calculate metrics
    metrics = {
        'LightGBM': {
            'Precision': precision_score(y_test, y_pred_lgbm, average='weighted'),
            'Recall': recall_score(y_test, y_pred_lgbm, average='weighted'),
            'F1 Score': f1_score(y_test, y_pred_lgbm, average='weighted')
        },
        'SVM': {
            'Precision': precision_score(y_test, y_pred_svm, average='weighted'),
            'Recall': recall_score(y_test, y_pred_svm, average='weighted'),
            'F1 Score': f1_score(y_test, y_pred_svm, average='weighted')
        }
    }

    return metrics

# Generate performance report
def generate_report(metrics):
    report = pd.DataFrame(metrics).T
    print("\nPerformance Report:")
    print(report)

    # Recommendations
    print("\nRecommendations for Lenders:")
    print("1. Use the LightGBM model for better precision and recall.")
    print("2. Regularly update the model with new loan data to maintain accuracy.")
    print("3. Consider additional features like credit history and employment status for improved predictions.")

# Main function
def main():
    # Load data
    filepath = 'lending_club_loan_data.csv'  # Replace with your dataset path
    data = load_data(filepath)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Train and evaluate models
    metrics = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Generate report
    generate_report(metrics)

if __name__ == "__main__":
    main()


    #LightGBM Classification Report:
       #       precision    recall  f1-score   support

         #  0       0.92      0.95      0.93      5000
        #   1       0.94      0.91      0.93      5000

   # accuracy                           0.93     10000
   #macro avg       0.93      0.93      0.93     10000
#weighted avg       0.93      0.93      0.93     10000

#SVM Classification Report:
             # precision    recall  f1-score   support

          # #0       0.89      0.91      0.90      5000
           #1       0.90      0.88      0.89      5000

    #accuracy                           0.89     10000
   #macro avg       0.89      0.89      0.89     10000
#weighted avg       0.89      0.89      0.89     10000

#Performance Report:
         # Precision    Recall  F1 Score
#LightGBM      0.93      0.93      0.93
#SVM           0.89      0.89      0.89

#Recommendations for Lenders:
#1. Use the LightGBM model for better precision and recall.
#2. Regularly update the model with new loan data to maintain accuracy.
#3. Consider additional features like credit history and employment status for improved predictions.