# src/train_model.py
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import traceback

from src.data_processing import load_csv, basic_clean, feature_engineer

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Script started")  # Add this at the top

def train(path, label_col='Class'):
    print("Loading CSV...")
    df = load_csv(path)
    print("Cleaning data...")
    df = basic_clean(df)
    print("Feature engineering...")
    df = feature_engineer(df)

    print("Preparing features and labels...")
    drop_cols = [label_col]
    X = df.drop(columns=drop_cols, errors='ignore')
    # If there are non-numeric cols, drop or encode (simple: drop for now)
    X = X.select_dtypes(include=['int64','float64'])
    y = df[label_col].astype(int)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_s, y_train)

    print("Training model...")
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
    model.fit(X_train_bal, y_train_bal)

    print("Predicting...")
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:,1]

    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    print("Saving model...")
    joblib.dump({'model': model, 'scaler': scaler, 'features': X.columns.tolist()}, f"{MODEL_DIR}/rf_bundle.joblib")

if __name__ == "__main__":
    try:
        train("data/creditcard1.csv", label_col='Class')
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
