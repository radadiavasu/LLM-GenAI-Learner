from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Flow
# 1. load
# 2. split
# 3. store vectors (embeddings) in vectordb
# 4. retrivers
# 5. func for Q & A

load_dotenv()
os.getenv("HF_TOKEN")

# load
# pdf_loader = PyPDFLoader("Chanakya-Neeti-Hindi.pdf")
pdf_loader = PyPDFLoader("2024_June_music_architectures.pdf")
docs = list(pdf_loader.lazy_load())

# split into small chunks (vectors / embeddings) and send it further for store them in vectordb.
spliter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=800)
doc_split = spliter.split_documents(documents=docs)

# store vectors (make vectordb)
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(doc_split, embed)

# retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# llm start

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    # huggingfacehub_api_token=hf_token,
)

client = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Question: {question}\n\nContent:\n{context}")
])

parser = StrOutputParser()
chain = prompt | client | parser


def rag_query(question: str):
    retrived_docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in retrived_docs])
    
    return chain.invoke({"question": {question}, "context": {context}})



def talk():
    while True:
        user_input = input("You: ").strip()
        if user_input in ['break', 'cls', 'close', 'exit', 'bye', 'q', 'quit']:
            print("Bye Bye, call me whenever you need...")
            break
        try:
            answer = rag_query(user_input)
            print(f"Chatbot: {answer}\n")
        except Exception as e:
            print(f"Something wrong, {e}\n")
            
            

if __name__ == "__main__":
    talk()