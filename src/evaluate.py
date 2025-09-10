# src/evaluate.py
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
from src.data_processing import feature_engineer  # Add this import

def load_model(path="models/rf_bundle.joblib"):
    bundle = joblib.load(path)
    return bundle['model'], bundle['scaler'], bundle['features']

def eval_on_test(df_path, label_col='Class'):
    model, scaler, features = load_model()
    df = pd.read_csv(df_path)
    df = feature_engineer(df)  # Apply feature engineering here
    X = df[features]
    y = df[label_col].astype(int)

    Xs = scaler.transform(X)
    y_prob = model.predict_proba(Xs)[:,1]
    y_pred = (y_prob > 0.5).astype(int)

    print(classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    print("Confusion matrix:\n", cm)

    # ROC
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    # Top flagged (highest probability)
    df['fraud_prob'] = y_prob
    flagged = df.sort_values('fraud_prob', ascending=False).head(20)
    print(flagged.head(10).to_string())

    

if __name__ == "__main__":
    eval_on_test("data/creditcard1.csv", label_col='Class')

