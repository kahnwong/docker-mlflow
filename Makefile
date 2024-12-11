start:
	mlflow server --port 8080

test:
	uv run python3 docker_mlflow/experiment_tracking.py
