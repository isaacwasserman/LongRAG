from llama_index.llms.bedrock import Bedrock
import boto3
import dotenv
import os

# Usage:
# from aws_mixtral import mixtral
# mixtral.complete(...) (same as using gpt)

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

mixtral = Bedrock(
    model="mistral.mixtral-8x7b-instruct-v0:1",
    aws_access_key_id=aws_access_key_id_temp,
    aws_secret_access_key=aws_secret_access_key_temp,
    aws_session_token=aws_session_token,
    region_name="us-east-1",
)
