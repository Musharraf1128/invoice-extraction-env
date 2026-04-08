from fastapi import FastAPI
from openai import OpenAI
import os

app = FastAPI()

API_BASE_URL = os.getenv("API_BASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

if not API_BASE_URL or not HF_TOKEN or not MODEL_NAME:
    raise ValueError("Missing required environment variables")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

@app.get("/")
def test_llm():
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say hello in one sentence"}]
        )
        return {"response": res.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}