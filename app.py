import streamlit as st
import pandas as pd
import joblib
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# =====================================
# PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="Automobile Market Analytics",
    layout="wide"
)

st.title("🚗 Automobile Market Analytics")
st.subheader("Fuel Type Classification Models")

# =====================================
# MODEL FILES
# =====================================

MODEL_PATHS = {
    "Logistic Regression":
        "models/logistic_regression.pkl",

    "Decision Tree":
        "models/decision_tree.pkl",

    "KNN":
        "models/knn.pkl",

    "Naive Bayes":
        "models/naive_bayes.pkl",

    "Random Forest":
        "models/random_forest.pkl"
}

# =====================================
# SELECT MODEL
# =====================================

selected_model = st.selectbox(
    "Select Classification Model",
    list(MODEL_PATHS.keys())
)

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

# =====================================
# PROCESS FILE
# =====================================

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    target_col = "Fuel_Type"

    if target_col not in data.columns:

        st.error(
            f"{target_col} column missing."
        )

        st.stop()

    X = data.drop(columns=[target_col])

    y = data[target_col]

    # =====================================
    # LOAD LABEL ENCODER
    # =====================================

    encoder = joblib.load(
        "models/label_encoder.pkl"
    )

    # Encode target variable
    #y_encoded = encoder.transform(y)

    # Create preview showing original and encoded values
    preview_df = pd.DataFrame({
        "Fuel_Type": encoder.inverse_transform(y),
        "Encoded_Value": y
    })

    st.write("Dataset Preview")
    st.dataframe(data.head())

    st.write("Fuel Type Label Encoding Mapping")
    st.dataframe(
        preview_df
        .drop_duplicates()
        .sort_values("Encoded_Value")
        .reset_index(drop=True)
    )

    # Use encoded target for evaluation
    #y = y_encoded.astype(int)
    # =====================================
    # LOAD MODEL
    # =====================================

    model_file = MODEL_PATHS[selected_model]

    model_object = joblib.load(model_file)

    if selected_model == "Naive Bayes":

        preprocessor = model_object["preprocessor"]
        model = model_object["model"]

        X_processed = preprocessor.transform(X)

        X_processed = X_processed.toarray()

        predictions = model.predict(
            X_processed
        )

        probabilities = model.predict_proba(
            X_processed
        )

    else:

        predictions = model_object.predict(
            X
        )

        probabilities = (
            model_object.predict_proba(X)
        )

    # =====================================
    # METRICS
    # =====================================

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y,
        predictions,
        average="weighted"
    )

    mcc = matthews_corrcoef(
        y,
        predictions
    )

    auc = roc_auc_score(
        y,
        probabilities,
        multi_class="ovr"
    )

    # =====================================
    # DISPLAY METRICS
    # =====================================

    st.header("Evaluation Metrics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    c2.metric(
        "AUC",
        f"{auc:.4f}"
    )

    c3.metric(
        "Precision",
        f"{precision:.4f}"
    )

    c1.metric(
        "Recall",
        f"{recall:.4f}"
    )

    c2.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    c3.metric(
        "MCC",
        f"{mcc:.4f}"
    )

    # =====================================
    # CLASSIFICATION REPORT
    # =====================================

    st.header("Classification Report")

    report = classification_report(
        y,
        predictions,
        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(report_df)

    # =====================================
    # CONFUSION MATRIX
    # =====================================

    st.header("Confusion Matrix")

    cm = confusion_matrix(
        y,
        predictions
    )

    fig, ax = plt.subplots(
        figsize=(8,6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")

    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # =====================================
    # ACTUAL VS PREDICTED
    # =====================================

    st.header(
        "Actual vs Predicted"
    )

    pred_labels = encoder.inverse_transform(
        predictions
    )

    actual_labels = encoder.inverse_transform(
        y
    )

    comparison = pd.DataFrame({
        "Actual": actual_labels,
        "Predicted": pred_labels
    })

    st.dataframe(
        comparison.head(50)
    )

else:

    st.info(
        "Upload generated test_data.csv"
    )