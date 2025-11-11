import mlflow
from mlflow.genai.scorers import Correctness, Guidelines
from openai import OpenAI

# start model via: <llama-server -hf ggml-org/gemma-3-1b-it-GGUF>

# ----- for inference -----
BASE_URL = "http://localhost:8080/v1"
API_KEY = ""
MODEL_NAME = "gemma3:270m"

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
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
## use gemini via litellm
## required env: `GEMINI_API_KEY`
mlflow.set_experiment("LLM Evaluation")
results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=predict_fn,
    scorers=[
        # Built-in LLM judge
        Correctness(model="gemini:/gemini-2.5-flash"),
        # Custom criteria using LLM judge
        Guidelines(
            model="gemini:/gemini-2.5-flash",
            name="is_english",
            guidelines="The answer must be in English",
        ),
    ],
)
