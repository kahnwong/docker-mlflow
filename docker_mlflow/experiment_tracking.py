import os

import dotenv
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


dotenv.load_dotenv()


################
# prep
################
iris = datasets.load_iris()

df = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)

X_train, X_test, y_train, y_test = train_test_split(
    df[
        [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
    ],
    df[["target"]],
    test_size=0.2,
    random_state=42,
)

params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

################
# mlflow
################
mlflow.set_tracking_uri(uri=os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("MLflow Quickstart")

with mlflow.start_run():
    mlflow.log_params(params)

    # log train data
    mlflow.log_table(data=X_train, artifact_file="training.json")
    mlflow.log_input(mlflow.data.from_pandas(X_train), context="training")

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # log validation data
    mlflow.log_table(data=y_train, artifact_file="validation.json")
    mlflow.log_input(mlflow.data.from_pandas(y_train), context="validation")

    # Predict on the test set
    y_pred = lr.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )
