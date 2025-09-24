from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
os.getenv("HF_TOKEN")

# pdf_load = PyPDFLoader('2024_June_music_architectures.pdf')
pdf_load = PyPDFLoader('Chanakya-Neeti-Hindi.pdf')

docs = list(pdf_load.lazy_load())

# print(docs[1].page_content)
# print(docs[1].metadata)

# this works for any type of chat huggingface model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    # huggingfacehub_api_token=hf_token,
)

client = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Answer the following question: {question}\n\nHere is some reference content:\n{context}")
])

parser = StrOutputParser()

chain = prompt | client | parser

finder = [d.page_content for d in docs]

result = chain.invoke({"question": "What is summary of second chapter in beginner friendly way?", "context": finder})

print(result)
# print(chain.get_graph().print_ascii())