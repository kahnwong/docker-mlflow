start:
	uv run mlflow server --port 8081

test:
	uv run python3 docker_mlflow/experiment_tracking.py
