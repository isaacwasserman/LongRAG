from llama_index.llms.bedrock import Bedrock
from llama_index.llms.openai import OpenAI
import boto3
import dotenv
import os
import tiktoken
import json

dotenv.load_dotenv()

aws_credentials = (
    boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_ACCESS_KEY_SECRET"),
        region_name="us-east-1",
    )
    .client("sts")
    .get_session_token()["Credentials"]
)

aws_session_token = aws_credentials["SessionToken"]
aws_access_key_id_temp = aws_credentials["AccessKeyId"]
aws_secret_access_key_temp = aws_credentials["SecretAccessKey"]

mixtral_llm = Bedrock(
    model="mistral.mixtral-8x7b-instruct-v0:1",
    aws_access_key_id=aws_access_key_id_temp,
    aws_secret_access_key=aws_secret_access_key_temp,
    aws_session_token=aws_session_token,
    region_name="us-east-1",
)

mistral_large_llm = Bedrock(
    model="mistral.mistral-large-2402-v1:0",
    aws_access_key_id=aws_access_key_id_temp,
    aws_secret_access_key=aws_secret_access_key_temp,
    aws_session_token=aws_session_token,
    region_name="us-east-1",
    context_size=32000,
)


class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        if model_name == "mixtral":
            self.model = mixtral_llm
            self.cost_per_input_token = 0.00000045
            self.cost_per_output_token = 0.0000007
        elif model_name == "mistral_large":
            self.cost_per_input_token = 0.000008
            self.cost_per_output_token = 0.000024
            self.model = mistral_large_llm
        elif model_name == "gpt4":
            self.model = OpenAI(temperature=0.2, model="gpt-4-0125-preview")
            self.cost_per_input_token = 0.00001
            self.cost_per_output_token = 0.00003
        else:
            raise ValueError("Model not found")

    def count_tokens(self, string):
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def log(self, prompt: str, completion: str):
        n_input_tokens = self.count_tokens(prompt)
        n_output_tokens = self.count_tokens(completion)
        cost = n_input_tokens * self.cost_per_input_token + n_output_tokens * self.cost_per_output_token
        with open(f"api_logs/{self.model_name}.jsonl", "a") as f:
            log_message = {
                "prompt": prompt,
                "completion": completion,
                "n_input_tokens": n_input_tokens,
                "n_output_tokens": n_output_tokens,
                "cost": cost,
            }
            f.write(json.dumps(log_message) + "\n")
        with open(f"api_logs/costs.json", "r") as f:
            costs = json.load(f)
        costs[self.model_name]["n_input_tokens"] += n_input_tokens
        costs[self.model_name]["n_output_tokens"] += n_output_tokens
        costs[self.model_name]["cost"] += cost
        with open(f"api_logs/costs.json", "w") as f:
            json.dump(costs, f)

    def complete(self, prompt: str):
        completion = self.model.complete(prompt)
        self.log(prompt, completion.text)
        return completion


mixtral = LLM("mixtral")
mistral_large = LLM("mistral_large")
gpt4 = LLM("gpt4")
