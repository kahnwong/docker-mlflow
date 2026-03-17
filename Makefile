start:
	uv run mlflow server --port 8081

tests:
	uv run python3 docker_mlflow/experiment_tracking.py
	uv run python3 docker_mlflow/llm_eval.py
