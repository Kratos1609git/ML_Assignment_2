import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
#import matplotlib.pyplot as plt
#import seaborn as sns

st.set_page_config(page_title="Cancer Risk Classification", layout="wide")

st.title("Cancer Risk Classification – ML Assignment 2")

# ============================
# Load models
# ============================

MODELS = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

#scaler = joblib.load("model/scaler.pkl")

# ============================
# Upload dataset
# ============================

uploaded_file = st.file_uploader(
    "Upload test CSV file (same structure as training data)",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    TARGET_COLUMN = "Cancer_Type"

    X = data.drop(columns=[TARGET_COLUMN])
    y_true = data[TARGET_COLUMN]

    # Encode target if needed
    if y_true.dtype == "object":
        y_true = y_true.astype("category").cat.codes

    model_name = st.selectbox("Select Model", list(MODELS.keys()))
    model = joblib.load(MODELS[model_name])

    # Scaling if required
    if model_name in ["Logistic Regression", "KNN"]:
        X_input = scaler.transform(X)
    else:
        X_input = X.values

    # Predictions
    y_pred = model.predict(X_input)
    y_prob = model.predict_proba(X_input)[:, 1]

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    st.subheader("Evaluation Metrics")
    st.write({
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "MCC": mcc
    })

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred))
