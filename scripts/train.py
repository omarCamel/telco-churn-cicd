from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data.csv"
MODEL_PATH = BASE_DIR / "models" / "champion.joblib"


def main():
    mlflow.set_experiment("telco_churn")

    df = pd.read_csv(DATA_PATH)
    target = "diagnosis"
    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe en el dataset.")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        col for col in X.columns if col not in numeric_features
    ]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
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

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )

    with mlflow.start_run(run_name="telco-churn-logreg") as run:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="binary", zero_division=0),
            "recall": recall_score(y_test, predictions, average="binary", zero_division=0),
            "f1": f1_score(y_test, predictions, average="binary", zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }

        mlflow.log_params({
            "test_size": 0.2,
            "random_state": 42,
            "strategy": "stratify",
            "model": "LogisticRegression",
            "target": target,
        })

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.log_artifact(str(DATA_PATH), artifact_path="dataset")
        mlflow.sklearn.log_model(model, artifact_path="model")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(model, MODEL_PATH)

        print(f"Run ID: {run.info.run_id}")
        print(f"Metrics: {metrics}")
        print(f"Modelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
