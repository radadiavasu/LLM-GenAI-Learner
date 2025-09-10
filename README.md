# Gen AI

## 1. Building Basic LLM Apps

-> Open Source Closed LLMs

-> Using LLM APIs

-> LangChain

-> HuggingFace

-> Ollama

## 2. Prompt Eng.

## 3. RAG

## 4. Fine Tuning

## 5. Agentic AI

## 6. LLMOps

## 7. Miscellanies

-------------------------------
# 1. LangChain: Its opensource framework with for developing LLM powered applications.

1. Fundamentals
2. RAG
3. Agents

-> Semantic Search: Search context aware text generation based or (embeddings)vector based to find the optimal output based on the query like for example: a pdf has information about a, b and c so thru semantic search it try to find some simlilarities from user query then answer according like if user asks about A then try to find similarities if high similarities found then get details and delivered.

* Components:

1. Models

-> Types of Models

1.1 Language models: It's LLM and with this we can make chatbots and AI agents.
1.2 Embedding models: Here they returns vector by sending texts and uses with semantic search.

2. Prompts: Prompts are basically you've providing to the LLM's.

3. Chains: you can also say chains are pipelines-for building LLM's for any of the task we build a chain wise blocks / flow to build a project.

3.1. parrel chain: It gives report format output means in detail answer given by input, which means the report generated with using multiple llm's.
3.2 conditional chains: Based on condition you can do different processings like, making AI agent who receive user feedback then given by user if feedback positive then send thank you to user otherwise send email to that customer feedback to the customer managing team.

4. Memory: LLM's API are stateless, here stateless means (each queries work independently like editable queries are not remembered about previous information that's the problem of stateless.) For example if you search anything like who is the Vladimir putin then LLM response you with information but LLM's API can't provide an information when you edit your query and ask about his age, so basically LLM API forget about previous info means they don't know what you've talking about.

-> Types of Memory

4.1 Conversation Buffer Memory: It stores you previous chats as a memory for certain threshold in DB but very expensive for storing large amount of data.
4.2 Conversation Window Buffer Memory: It keeps the last N interactions or you can say instead of saving large chunks of queries set in DB.
4.3 Summarize-Based Memory: It stores history from all of the chat it saves lots of texts and saves money too.
4.4 Custom Memory: Stores very relevant information like user's preferences and important key facts, also private information in custom memory.

5. Indexes: Contains document loader, text splitter, vector store, retrievers. Indexes are connect with your external information and can able to with Q & A, explaining , or take suggestions, improvements or modify etc. For ex. chatgpt solve your coding performance by enhancing your code structure while you attach your code file so it learns from what you have then debug the issue and solve it.

6. Agents: With help of LLM we can make AI agents. AI agents are modern version of traditional chatbots where normally chatbots just able to answer user's query whereas agents gives answers on custom data in semantic search + embeddings (vector) and NLU which performs well on non-linear data and able to answer accurately.

* Components for pdf based Q & A chatbot task:
1. Data Loader
2. Text Splitter
3. Embeddings (Vectors)
4. Database (MLOps)
5. Retrieval
6. Send to LLM

* With Langchain we can make (applications) like,

1. Conversational Chatbots
2. AI knowledge Assistanace ex. Live class 
3. AI Agents (Not just talk they work for you like ticket booking, calling, sending messages etc...)
4. Workflow Automation (Use interanlly for boost productivity in your company or schools  institute or home (for personal use)
5. Summerization / Research papers

Alternatives of Langchain

1. llamaindex
2. haystack
