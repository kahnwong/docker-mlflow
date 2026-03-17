import os

import mlflow
from mlflow.genai.scorers import Correctness, Guidelines
from openai import OpenAI

# ----- for inference -----
MODEL_NAME = "claude-haiku-4-5"

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ----- qa dataset -----
dataset = [
    {
        "inputs": {"question": "Can MLflow manage prompts?"},
        "expectations": {"expected_response": "Yes!"},
    },
    {
        "inputs": {"question": "Can MLflow create a taco for my lunch?"},
        "expectations": {
            "expected_response": "No, unfortunately, MLflow is not a taco maker."
        },
    },
]


# ----- prediction function -----
def predict_fn(question: str) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": question}],
    )

    return completion.choices[0].message.content


# ----- evaluation -----
mlflow.set_experiment("LLM Evaluation")

judge_model = f"openai:/{MODEL_NAME}"
results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=predict_fn,
    scorers=[
        # Built-in LLM judge
        Correctness(model=judge_model),
        # Custom criteria using LLM judge
        Guidelines(
            model=judge_model,
            name="is_english",
            guidelines="The answer must be in English",
        ),
    ],
)
