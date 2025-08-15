import os

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, median_absolute_error
from sklearn.model_selection import train_test_split

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
    "random_state": 8888,
}

################
# mlflow
################
mlflow.set_tracking_uri(uri=os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("MLflow Quickstart")

with mlflow.start_run():
    mlflow.set_tag("Training Info", "Basic LR model for iris data")
    mlflow.log_params(params)

    # log train data
    mlflow.log_table(data=X_train, artifact_file="training.json")
    mlflow.log_input(mlflow.data.from_pandas(X_train), context="training")

    # log validation data
    mlflow.log_table(data=y_train, artifact_file="validation.json")
    mlflow.log_input(mlflow.data.from_pandas(y_train), context="validation")

    # train
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train["target"])

    # log model
    model_name = "iris_model"
    signature = infer_signature(X_train, lr.predict(X_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        name=model_name,
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )

    # evaluate
    ## predict against test set
    y_pred = lr.predict(X_test)

    ## log accuracy
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    mae = median_absolute_error(y_test, y_pred)
    mlflow.log_metric("mae", accuracy)
