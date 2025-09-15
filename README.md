# Gen AI Learning Roadmap  

## 1. Building Basic LLM Apps  
- Open Source & Closed LLMs  
- Using LLM APIs  
- LangChain  
- HuggingFace  
- Ollama  

## 2. Prompt Engineering  

## 3. Retrieval-Augmented Generation (RAG)  

## 4. Fine-Tuning  

## 5. Agentic AI  

## 6. LLMOps  

## 7. Miscellaneous  

---

# LangChain Overview  

LangChain is an **open-source framework** for developing applications powered by Large Language Models (LLMs).  
It provides tools to build scalable, production-ready AI systems.  

### Core Areas  
1. Fundamentals  
2. RAG (Retrieval-Augmented Generation)  
3. Agents  

---

## 🔎 Semantic Search  
Semantic search enhances context-aware text generation using **embeddings (vector representations)**.  

**Example:**  
A PDF contains information about topics A, B, and C. If a user asks about A, semantic search finds similarities with the embeddings and retrieves the most relevant details.  

---

## 🧩 Components of LangChain  

### 1. Models  
- **Language Models** → Standard LLMs used for chatbots and AI agents.  
- **Embedding Models** → Convert text into vectors for semantic search and retrieval.  

### 2. Prompts  
Instructions or inputs you provide to LLMs to guide their responses.  

### 3. Chains  
Chains are pipelines that connect multiple components together to perform tasks.  

- **Parallel Chains** → Generate detailed, report-style answers using multiple LLMs.  
- **Conditional Chains** → Perform different actions based on conditions (e.g., positive/negative feedback workflows).  

### 4. Memory  
LLM APIs are **stateless** (they don’t remember past queries). Memory helps overcome this limitation.  

- **Conversation Buffer Memory** → Stores complete conversation history (expensive for large data).  
- **Conversation Window Memory** → Keeps only the last N interactions.  
- **Summary-Based Memory** → Summarizes history, saving cost and space.  
- **Custom Memory** → Stores personalized or private key facts (e.g., user preferences).  

### 5. Indexes  
Indexes connect external information to the LLM. They include:  
- Document Loader  
- Text Splitter  
- Vector Store  
- Retrievers  

**Use Case Example:**  
Attach a code file → LangChain indexes it → AI suggests improvements or fixes bugs. (Just like we do regularly with chatgpt.)

### 6. Agents  
Agents are advanced AI assistants built on LLMs.  
- Unlike simple chatbots, agents combine **semantic search + embeddings + NLU** to handle complex, non-linear queries.  
- They can not only answer but also **act** (e.g., book tickets, send emails).  

---

## ⚙️ Example: PDF-based Q&A Chatbot  
Components needed:  
1. Data Loader  
2. Text Splitter  
3. Embeddings (Vectors)  
4. Database (MLOps)  
5. Retrieval  
6. LLM  

---

## 🚀 Applications with LangChain  
- Conversational Chatbots  
- AI Knowledge Assistants (e.g., live class helper)  
- AI Agents (performing real tasks like booking tickets, sending messages)  
- Workflow Automation (productivity tools for companies, schools, or personal use)  
- Summarization / Research Support  

---

## 🔄 Alternatives to LangChain  
- **LlamaIndex**  
- **Haystack**  

---
# Components

## 1. Models  

### 1.1 Language Models  
Language Models (LLMs) are used for generating and understanding text. With them, we can build chatbots, AI assistants, and other NLP-powered applications.  

#### Types of Language Models  

1. **LLMs (Base Models)**  
   - Used for general-purpose tasks like text generation, summarization, Q&A, and code completion.  
   - Do not have built-in memory.  
   - No real understanding of users — every request is treated independently.  

2. **Chat Models**  
   - Designed for conversational tasks.  
   - Take a sequence of messages as input and return chat-like responses.  
   - Support conversational memory (e.g., remembering context within a session).  

---

#### Important Parameters  

- **Temperature**: Controls randomness in model outputs.  
  - Lower values (`0.0 – 0.3`) → more deterministic and predictable (factual answers).  
  - Mid values (`0.5 – 0.7`) → balanced, natural responses (general Q&A, explanations).  
  - Higher values (`0.9 – 1.2`) → more creative and diverse (stories, jokes, brainstorming).  
  - Very high (`1.2+`) → wild randomness, often incoherent.  

- **max_completion_tokens**: Defines the maximum number of tokens (roughly words/pieces of words) the model can generate.  
  - Useful for **cost control** since API usage is charged per token.  
  - Setting tokens too high wastes quota quickly.  

---

#### Open vs Closed Source Models  

- **Closed-source models (paid)**:  
  Examples: OpenAI (GPT), Anthropic (Claude), Google (Gemini).  
  - Access via paid APIs.  

- **Open-source models (free/community-driven)**:  
  Examples: Hugging Face models, Transformers, LLaMA, Mistral.  
  - Can be used via hosted APIs (may have free limits, then paid).  
  - Or downloaded and run locally (requires good hardware).  

---

### 1.2 Embedding Models  
Embedding models convert text into **vectors (arrays of numbers)**.  
These vectors capture semantic meaning and are mainly used for **semantic search and RAG (Retrieval-Augmented Generation)** applications.  

#### Example:  
Suppose you have documents about cricketers A, B, C, D, and E.  

1. You load an embedding model and embed both documents and a user query (e.g., *“Who is A?”*).  
2. The model outputs vectors (high-dimensional numbers).  
3. Semantic search compares the query vector with document vectors.  
4. The closest match (highest similarity score) is retrieved.  
5. The LLM then uses this retrieved context to generate an accurate answer.  

---

#### Embedding Parameters  

- **Dimensions**: The size of the vector (e.g., 384, 768, 1536).  
  - Fewer dimensions = cheaper & faster, but capture less context.  
  - More dimensions = richer context, but higher cost and memory usage.  

---

#### Summary  

- **Language Models** return **text outputs** from text inputs.  
- **Embedding Models** return **vectors (numbers)** from text inputs.  
- Embeddings are crucial for improving **semantic search**, **RAG**, and building context-aware applications.  


-------------------------------------------------------------------------------------------------------------------------------------------------------------
Prompts: Prompts are the input instruction or a query asked by user to the model to guide its output.

Types of prompts:

1. Text based prompts:

To make prompt template use langchain's built-in PromptTemplate class and give your template.

-> Single Message(single turn stand alone queries)

* Static message: Those templates are heavily reliable & sensitive on LLM's, with some minor changes will make huge impact in overall response and not preferred, now why? For example you want to find something and just have the topic idea but not the related name like "I want to find this type of research but don't know the name" so at that time LLM gives you very wrong output which is not desirable.

* Dynamic message: These prompts are widely accepts because they have the solid structure of prompts and we called it "prompt templates" now here we set template to take some additional inputs with same example you give some topic idea or summarize the topic then the style of the topic and the explanation style so now this makes the very unique thing with dynamic prompts so the real world example is, chatgpt at last it asks to modify or style or extend or make it simple etc... once you asks something to chatgpt so basically OpenAI set the dynamic templates and try to dig in that specific topic you've asked.

Prompt Template: In langchain for create creative prompts dynamically by inserting variables into a predefined template. Instead of hardcoding template PromptTemplate allows us to define placeholders that can be use with different inputs. This makes easy to implement, reusable, flexible and easy to manage especially while working with the dynamic user inputs.

Q Why should I need to make PromptTemplate for over "f" strings?
Ans: There are three reasons,

*. Default Validation: It provide validations which really help full to prevent any kind of errors while developing not on server. For example If you provide three variables in template but mention only two or entered wrong before invoke so it throws the error on developing side not on server side.

*. Reusable: Eventually you can reuse template with some other types of tasks too. It saves lots of time.

*. LangChain Ecosystem: Generally here you can make the json file and saves your each time template in professions way and that is ecosystem you make for any kind of tasks like Make a separate file which can generate your each type of prompts template in json format and make a file then use into your main file. So basically ecosystem provide robustness while dealing with heavy or complex tasks.

-> List of Messages (Multi-turn conversation)

(Classes for static messages in langchain)

1. System Message: It is the asking when starting conversations like you're a very helpful, you're such a knowledgeable person, Hi, How are you etc.....

2. User Message: It is basically we (humans) send input to the LLM.

3. AI Message: It is the response (answer) messages we get from AI (LLM).

(Classes for dynamic messages in langchain)

* Chat Prompt Template: Almost same as "Prompt Template" just different is that you don't decide the specific type of message example for user message "You've good taste in {domain}, here domain is not static it could be any domain so it's dynamic same for human message "Give me the info about this {topic}, topic could be anything."

* Message Placeholder: This is special placeholder use inside "chat prompt template" to dynamically insert chat history or a list of messages at runtime.

2. Image / Sound / Video Prompts (Multi model prompts):
