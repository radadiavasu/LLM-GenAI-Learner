from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

hf_token = load_dotenv("HF_TOKEN")

# this works for any type of chat huggingface model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    huggingfacehub_api_token=hf_token,
)


client = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

parser = StrOutputParser()

chain = prompt | client | parser

result = chain.invoke({"question": "what is the difference between langchain and langgraph"})

print(result)