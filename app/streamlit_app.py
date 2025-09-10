# app/streamlit_app.py
import sys
import os

# Force add project root (FraudDetection) to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Python Path:", sys.path)  # Debugging line

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src.data_processing import basic_clean, feature_engineer


@st.cache_data
def load_bundle():
    return joblib.load("models/rf_bundle.joblib")

st.title("Fraud Detection - Demo Dashboard")
bundle = load_bundle()
model = bundle['model']
scaler = bundle['scaler']
features = bundle['features']

uploaded = st.file_uploader("Upload transaction CSV (or use data/creditcard.csv)", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("data/creditcard1.csv")

df = basic_clean(df)
df = feature_engineer(df)
X = df[features].select_dtypes(include=['int64','float64'])

Xs = scaler.transform(X)
probs = model.predict_proba(Xs)[:,1]
df['fraud_prob'] = probs

st.metric("Dataset rows", len(df))
st.metric("Predicted frauds (prob > 0.5)", (df['fraud_prob']>0.5).sum())

st.subheader("Top flagged transactions")
st.dataframe(df.sort_values("fraud_prob", ascending=False).head(20))

st.subheader("Fraud distribution by hour (if hour present)")
if 'hour' in df.columns:
    chart = df.groupby('hour')['fraud_prob'].mean()
    st.line_chart(chart)
