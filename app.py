import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load data
df = pd.read_csv("/content/fraudTest.csv")

print(df.columns)
print(df.head())


# Separate features and target
X = df.drop(['is_fraud', 'trans_date_trans_time', 'first', 'last', 'gender', 'street', 'city',
             'state', 'job', 'dob', 'merchant', 'category', 'trans_num'], axis=1)
y = df['is_fraud']# 1 = Fraud, 0 = Legit

# Drop rows where y has NaN values
# This ensures that train_test_split can process the target variable without issues
# especially with stratify=y
nan_rows = y.isnull()
X = X[~nan_rows]
y = y[~nan_rows]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "fraud_detector.pkl")

def predict_transaction(transaction_data):
    """
    transaction_data: list or numpy array with the same number of features as X (excluding 'Class')
    """
    model = joblib.load("fraud_detector.pkl")
    prediction = model.predict([transaction_data])
    return "Fraudulent Transaction!" if prediction[0] == 1 else "Legitimate Transaction."
