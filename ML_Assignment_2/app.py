import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
#!pip install matplotlib
#!pip install seaborn
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Cancer Risk Classification", layout="wide")
st.title("Cancer Risk Classification – ML Assignment 2")

MODELS = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    TARGET_COLUMN = "Cancer_Type"

    X = data.drop(columns=[TARGET_COLUMN])
    y_true = data[TARGET_COLUMN]

    if y_true.dtype == "object":
        y_true = y_true.astype("category").cat.codes

    model_name = st.selectbox("Select Model", list(MODELS.keys()))
    model = joblib.load(MODELS[model_name])

    # Scale ONLY for LR and KNN
    if model_name in ["Logistic Regression", "KNN"]:
        scaler = StandardScaler()
        X_input = scaler.fit_transform(X)
    else:
        X_input = X.values

    y_pred = model.predict(X_input)
    y_prob = model.predict_proba(X_input)[:, 1]

    st.subheader("Evaluation Metrics")
    st.write({
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    })

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred))
