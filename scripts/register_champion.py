
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    experiment = mlflow.get_experiment_by_name("Telco_Churn")
    if experiment is None:
        raise ValueError("No existe el experimento 'Telco_Churn'. Ejecuta primero python scripts/train.py")

    client = MlflowClient(tracking_uri=TRACKING_URI)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No se encontraron runs para registrar el Champion.")

    winner = runs[0]
    run_id = winner.info.run_id
    model_uri = f"runs:/{run_id}/model"

    registered_name = "TelcoChurnClassifier"
    result = mlflow.register_model(model_uri=model_uri, name=registered_name)
    version = result.version
    print(f"Modelo registrado: {registered_name} v{version}")

    client.set_registered_model_alias(
        name=registered_name,
        alias="champion",
        version=version,
    )
    print(f"Alias 'champion' asignado a {registered_name} v{version}")

    model = mlflow.pyfunc.load_model(f"models:/{registered_name}@champion")
    print("Modelo champion cargado correctamente desde MLflow.")
    return model


if __name__ == "__main__":
    main()
