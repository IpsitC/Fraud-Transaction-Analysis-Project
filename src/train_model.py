# src/train_model.py
import os
import joblib
import traceback
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

from src.data_processing import load_csv, basic_clean, feature_engineer

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Script started")

def train(path, label_col='Class'):
    print("Loading CSV...")
    df = load_csv(path)
    print("Cleaning data...")
    df = basic_clean(df)
    print("Feature engineering...")
    df = feature_engineer(df)

    print("Preparing features and labels...")
    X = df.drop(columns=[label_col], errors='ignore')
    X = X.select_dtypes(include=['int64','float64'])  # numeric only
    y = df[label_col].astype(int)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_s, y_train)

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # Store results for summary
    results = []

    for name, model in models.items():
        print(f"\n===== Training {name} =====")
        model.fit(X_train_bal, y_train_bal)

        print("Predicting...")
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]

        print("Classification report:")
        print(classification_report(y_test, y_pred, digits=4))
        roc_auc = roc_auc_score(y_test, y_proba)
        print("ROC AUC:", roc_auc)

        # Save metrics for summary table
        results.append({
            "Model": name,
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc
        })

        # Save model
        model_path = f"{MODEL_DIR}/{name}_bundle.joblib"
        joblib.dump({'model': model, 'scaler': scaler, 'features': X.columns.tolist()}, model_path)
        print(f"Saved {name} model to {model_path}")

    # Print summary table
    print("\n===== Summary of All Models =====")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    # 🔹 Save summary to CSV
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    summary_csv_path = os.path.join(results_dir, "model_results.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved summary results to {summary_csv_path}")

    # 🔹 Plot comparison chart
    import matplotlib.pyplot as plt

    metrics = ["Precision", "Recall", "F1-score", "ROC-AUC"]
    summary_df.set_index("Model")[metrics].plot(kind="bar", figsize=(10,6))

    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(title="Metrics")
    plt.xticks(rotation=0)
    plt.tight_layout()

    chart_path = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved comparison chart to {chart_path}")


if __name__ == "__main__":
    try:
        train("data/creditcard1.csv", label_col='Class')
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()


