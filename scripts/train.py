import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data.csv"
MODEL_PATH = BASE_DIR / "models" / "champion.joblib"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop(columns=[col for col in cleaned.columns if cleaned[col].isna().all()], errors="ignore")
    if "id" in cleaned.columns:
        cleaned = cleaned.drop(columns=["id"])
    return cleaned


def build_model(name: str, params: dict):
    numeric_features = [
        col for col in params["feature_frame"].columns if pd.api.types.is_numeric_dtype(params["feature_frame"][col])
    ]
    categorical_features = [
        col for col in params["feature_frame"].columns if col not in numeric_features
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    estimator = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
    }[name]

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


def train_and_log_model(model_name: str, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average="binary", zero_division=0),
            "recall": recall_score(y_test, preds, average="binary", zero_division=0),
            "f1": f1_score(y_test, preds, average="binary", zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }

        mlflow.log_params({
            "model": model_name,
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
            "target": "diagnosis",
        })

        if model_name == "LogisticRegression":
            mlflow.log_params({"C": 1.0, "max_iter": 1000})
        elif model_name == "RandomForest":
            mlflow.log_params({"n_estimators": 200, "max_depth": None})
        elif model_name == "SVM":
            mlflow.log_params({"C": 1.0, "gamma": "scale", "kernel": "rbf"})

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.log_artifact(str(DATA_PATH), artifact_path="dataset")
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        print(f"{model_name} | run_id={run.info.run_id} | metrics={metrics}")
        return run.info.run_id


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("Cancer_Diagnosis")

    df = pd.read_csv(DATA_PATH)
    df = clean_dataset(df)

    target = "diagnosis"
    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe en el dataset.")

    X = df.drop(columns=[target])
    y = df[target].map({"B": 0, "M": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_names = ["LogisticRegression", "RandomForest", "SVM"]
    trained_runs = []

    for model_name in model_names:
        model = build_model(model_name, {"feature_frame": X})
        run_id = train_and_log_model(model_name, model, X_train, X_test, y_train, y_test)
        trained_runs.append((model_name, run_id))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\nRuns MLflow registrados:")
    for model_name, run_id in trained_runs:
        print(f"- {model_name}: {run_id}")


if __name__ == "__main__":
    main()
