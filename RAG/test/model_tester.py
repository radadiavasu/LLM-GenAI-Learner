# import os
from openai import OpenAI
from dotenv import load_dotenv

hf_token = load_dotenv("HF_TOKEN")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token,
    # api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    # model="meta-llama/Llama-3.2-3B-Instruct:novita",
    model="meta-llama/Llama-3.1-8B-Instruct:nebius",
    messages=[
        {
            "role": "user",
            "content": "What are runnables including why, what and types of runnables in langchain?"
        }
    ],
)

print(completion.choices[0].message.content)