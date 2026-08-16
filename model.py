import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

# =======================================
# Create folders
# =======================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =======================================
# Load dataset
# =======================================

df = pd.read_csv("dataset/automobile_market_analytics.csv")

print("Dataset Shape:", df.shape)

# =======================================
# Target Column
# =======================================

TARGET = "Fuel_Type"

# Remove rows with missing target
df = df.dropna(subset=[TARGET])

# Features and Target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# =======================================
# Encode target
# =======================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

# =======================================
# Train Test Split
# =======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =======================================
# Save Test Dataset
# =======================================

test_df = X_test.copy()
test_df["Fuel_Type"] = y_test

test_df.to_csv(
    "test_data.csv",
    index=False
)

# =======================================
# Preprocessing
# =======================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

numeric_transformer = Pipeline(
    [
        ("imputer",
         SimpleImputer(strategy="median")),

        ("scaler",
         StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    [
        ("imputer",
         SimpleImputer(strategy="most_frequent")),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    [
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

# =======================================
# Models
# =======================================

models = {
    "logistic_regression":
        LogisticRegression(
            max_iter=5000
        ),

    "decision_tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "knn":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "naive_bayes":
        GaussianNB(),

    "random_forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}

results = []

# =======================================
# Train Models
# =======================================

for model_name, model in models.items():

    print(f"\nTraining {model_name}")

    if model_name == "naive_bayes":

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()

        model.fit(
            X_train_processed,
            y_train
        )

        y_pred = model.predict(
            X_test_processed
        )

        y_prob = model.predict_proba(
            X_test_processed
        )

        joblib.dump(
            {
                "preprocessor": preprocessor,
                "model": model
            },
            f"models/{model_name}.pkl"
        )

    else:

        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "classifier",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        y_pred = pipeline.predict(
            X_test
        )

        y_prob = pipeline.predict_proba(
            X_test
        )

        joblib.dump(
            pipeline,
            f"models/{model_name}.pkl"
        )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr"
    )

    results.append([
        model_name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ])

# =======================================
# Save Metrics
# =======================================

metrics_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]
)

metrics_df.to_csv(
    "outputs/metrics.csv",
    index=False
)

print("\n========== MODEL RESULTS ==========")
print(metrics_df)
print("===================================")