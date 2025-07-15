import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException
import mlflow.sklearn
import dagshub
import logging

dagshub.init(repo_owner='User_SH_ML', repo_name='MLflow', mlflow=True)

# Set up logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Dataset URL
csv_url = "https://raw.githubusercontent.com/aniruddhachoudhury/Red-Wine-Quality/master/winequality-red.csv"

# Connect to remote server (Dagshub)
remote_server_uri = "https://dagshub.com/User_SH_ML/MLflow.mlflow"
mlflow.set_tracking_uri(remote_server_uri)

def eval_metrics(actual, pred):
    """Calculates evaluation metrics: RMSE, MAE, R2_Score."""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    try:
        # Load the dataset
        # CHANGED: Attempting with comma as separator based on the error output
        # If this still fails, the issue is more complex (e.g., file corruption, specific environment setup)
        data = pd.read_csv(csv_url, sep=',')
        # --- Diagnostic Prints ---
        print("DataFrame head:\n", data.head())
        print("Columns in DataFrame:", data.columns.tolist())
        # --- End Diagnostic Prints ---
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection and the URL: %s", csv_url
        )
        sys.exit(1)

    # --- Added Check for 'quality' column ---
    # The expected column name should be 'quality' if read correctly.
    # If the file is indeed semicolon separated and the previous 'sep=;' failed,
    # then the entire line was read as one column, including the literal string 'quality' as part of that column name.
    # By changing to sep=',', if the *actual* file (what pandas is reading) uses commas,
    # then 'quality' should appear as its own column.
    if "quality" not in data.columns:
        logger.error(f"Error: 'quality' column not found in the dataset. Available columns: {data.columns.tolist()}")
        # If it still fails with ',', it implies that the single column
        # actually contains the whole string and 'quality' isn't split out.
        # In this specific case, the problem is that pandas is not splitting columns.
        # The ultimate fallback if the file is truly semicolon-separated and pandas fails with `sep=';'`
        # is to explicitly read the file line by line and parse it, but that's a last resort.
        sys.exit(1)
    # --- End Added Check ---

    # Separate features and target
    X = data.drop(["quality"], axis=1)
    y = data["quality"]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameters for ElasticNet
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(X_train, y_train)

        predicted_qualities = lr.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
        print(f"  RMSE: {rmse:f}")
        print(f"  MAE: {mae:f}")
        print(f"  R2: {r2:f}")

        # Log parameters, metrics, and the model
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Infer signature and log the model
        predictions_for_signature = lr.predict(X_train)
        signature = infer_signature(X_train, predictions_for_signature)

        if tracking_url_type_store != "file":
            try:
                mlflow.sklearn.log_model(
                    sk_model=lr,
                    artifact_path="model",
                    signature=signature,
                    input_example=X_train.sample(1),
                    registered_model_name="ElasticnetWineModel"
                )
            except MlflowException as e:
                logger.error(f"Failed to log model with registered_model_name (possibly already exists or permission issue): {e}. Logging without registered_model_name.")
                mlflow.sklearn.log_model(
                    sk_model=lr,
                    artifact_path="model",
                    signature=signature,
                    input_example=X_train.sample(1)
                )
        else:
            mlflow.sklearn.log_model(
                sk_model=lr,
                artifact_path="model",
                signature=signature,
                input_example=X_train.sample(1)
            )

        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")