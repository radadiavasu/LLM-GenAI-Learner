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


---
# Prompts  

Prompts are the **instructions or queries** given by a user to an LLM in order to guide its output.  
A well-designed prompt can dramatically improve the quality, accuracy, and relevance of the response.  

---

## Types of Prompts  

### 1. Text-Based Prompts  

#### ➝ Single Message (Single-turn, stand-alone queries)  

- **Static Messages**  
  - Rigid templates that are heavily dependent on exact wording.  
  - Small changes in phrasing = very different results means very sensitive and highly reliable on LLM.  
  - **Not preferred** for exploratory or uncertain queries.  
  - **Example**: Asking *“I want this type of research but don’t know its name”* → The model may hallucinate or give irrelevant answers.  

- **Dynamic Messages**  
  - More reliable and flexible.  
  - Built with **Prompt Templates** that allow placeholders for variables.  
  - Useful when users don’t know exact names but can provide partial info, context, or desired output style.  
  - **Example**: Instead of hardcoding, you define a template:  
    ```
    Summarize the topic "{topic}" in the style of {style}.
    ```  
    If the user provides *topic = AI in Healthcare* and *style = simple*, the model generates a tailored response.  

---

#### PromptTemplate (LangChain)  

LangChain provides the `PromptTemplate` class to create structured, reusable, and validated prompts.  
This is preferred over basic Python `f-strings`.  

**Why use PromptTemplate over f-strings?**  

1. **Validation**  
   - Prevents runtime errors.  
   - Example: If you define 3 variables but provide only 2, LangChain throws an error during development instead of failing silently at runtime.  

2. **Reusability**  
   - Templates can be used across different tasks with minimal changes.  
   - Saves time and avoids repetitive code.  

3. **Ecosystem Support**  
   - Prompts can be stored in JSON or separate config files.  
   - Encourages building a clean, maintainable **prompt ecosystem** for complex projects.  

---

### 2. List of Messages (Multi-turn Conversations)  

LangChain also supports **multi-message conversations** with specialized classes.  

#### Classes for Static Messages  
- **System Message** → Defines behavior of the assistant.  
  *Example: “You are a helpful and knowledgeable assistant.”*  
- **User Message** → Input provided by the user.  
- **AI Message** → The response returned by the AI.  

- Example:
  
```python
 from langchain_core.prompts import ChatPromptTemplate, HumanMessage, SystemMessage

 chat_template = ChatPromptTemplate.from_messages(
     [
         SystemMessage(content="You are a helpful AI assistant."),
         HumanMessage(content="What is the capital of {country}?"),
     ]
 )
 formatted_prompt = chat_template.format_messages(country="France")
 # formatted_prompt will be a list of message objects:
 # [SystemMessage(content='You are a helpful AI assistant.'), HumanMessage(content='What is the capital of France?')]
```

#### Classes for Dynamic Messages  
- **ChatPromptTemplate**  
  - Similar to `PromptTemplate` but used for conversations chat-style modern LLM's.
  - Allows variables inside user or AI messages.  
  - Example:  
    ```
    Human: "Give me information about {topic}"
    AI: "Sure! Here's a summary of {topic}..."
    ```  

- **MessagePlaceholder**  
  - Special placeholder used inside a `ChatPromptTemplate` to dynamically insert **chat history** or **lists of messages** at runtime.  
  - Ensures the model has context across turns.  

## What is the difference between PromptTemplate and ChatPromptTemplate?
- Ans is PromptTemplate is generally used for regular text-based (language) models like davnici 003 or old gpt models and these are non-structured messages because it gives one big block of text on other hand ChatPromptTemplate use in chat-based models (like ChatGPT, Anthropic, Gemini and other latest open source models) and these are roles-based models (`System`, `Human`, `AI`) makes it proper structurized in simple words you assign a roles. So if your model expect single string as a output then go for PromptTemplate otherwise ChatPromptTemplate to working with chat-models for making role-based AI agents.

---

### 3. Multi-Modal Prompts  
Prompts are not limited to text. Modern LLMs can handle:  
- **Image Prompts**  
- **Audio Prompts**  
- **Video Prompts**  

These are essential in **multi-modal AI systems** (e.g., asking an AI to describe an image or generate a caption for a video).  

---

## Summary  

- Prompts are **instructions to guide LLM output**.  
- **Static prompts** are fragile → avoid for real-world apps.  
- **Dynamic prompts with PromptTemplate** are powerful → reusable, validated, and ecosystem-friendly.  
- **Multi-turn prompts** allow conversational context.  
- **Multi-modal prompts** extend LLMs beyond text.  

---
# Structured Outputs: 

- We notice currently we get LLM's response (outputs) are in a unstructured format. Like look at the example

- 🔹 **LLM's un-structured output:**
```
User: Can we plan a trip for Paris?

-> LLM's unstructured output:

Here's a suggested planning, Morning: Visit the Eiffel tower,

Afternoon: Walk thru the Laure museum,

Evening: Enjoy dinner at a Seine riverside cafe.
```

- 🔹 **LLM's expected structured output:**

```json
[
{"Time": "Morning", "Activity": "Visit the Eiffel tower"}
{"Time": "Afternoon", "Activity": "Walk thru the Laure museum"}
{"Time": "Evening", "Activity": "Enjoy dinner at a Seine riverside cafe"}
]
```

## 🔹Q. Now why would we need structured output?

- Ans: Generally structured output will return well-defined data format (example: JSON) rather than free-form text. Now this makes really useful to work when integrate LLM's with other systems too.

- **Data Extraction:** Store structured data (e.g., JSON) directly in databases.

- **API Building:** For building API's from structured data so anyone can use across the world.

- **Agents:** AI agents need structured data for accurate actions.

## 🔹How many types of LLM's can generate structured output?

- Some LLMs (e.g., OpenAI, Google, Anthropic) provide structured output options:

1. `with_structured_output()` → Built-in function to enforce structured responses

2. `output_parser()` → For models that don’t support strict structure natively

- Methods (param) of `with_structured_output` for generate structured output,
-  1. 'JSON mode' (while working with claude or jemini because they only support json format)
   2. 'function calling' (while working with openai for calling function (agents)).

## 🔹 Data Formats for Structured Outputs

### 1. TypedDict (Built-in): Define keys & values with type hints relevent for prevents runtime errors (helps developers know expected structure) but no strict validation (e.g., if you expect int but get str, it won’t complain)

- 🔹 **Example**

```python
from typing import TypedDict

class TripPlan(TypedDict):
    time: str
    activity: str
```

### 2. Pydantic (Popular with FastAPI): Provide strong data validation & parsing and ensures the data you work with is correct, structured and safe-search.

- 🔹 **Example**

```python
from pydantic import BaseModel

class TripPlan(BaseModel):
    time: str
    activity: str
```

### 3. json_schema: Works across all programming languages (If you different language in front-end and different in back-end) so the it format the data in the common way.

- 🔹 **Example**

```json
{
  "type": "object",
  "properties": {
    "time": {"type": "string"},
    "activity": {"type": "string"}
  },
  "required": ["time", "activity"]
}
```

---
# Chains

### When building generative AI applications, instead of manually creating prompts, invoking the model, and handling responses step-by-step, Chains allow you to automate this pipeline. Chains simplify the process — you provide the input, trigger the chain, and the rest is handled automatically.

**For visualize the actual chain flow you can use:**

```python
chain.get_graph().print_ascii()
```

## Types of chains

`1. Simple Chain`: Normal chain as we make regularly like ```prompt | model | parser in chain.```

`2. Sequential Chain`: Used for performing sequential tasks like for ex, I have a input and want to generate the detailed report then from that report want to get summary, so clearly we want multiple LLM's and so for that we use like this make two templates one for detailed report on topic and another for the summary now take parser and make chain then invoke it and get the result.

`3. Parallel Chain`: We can use this for generate multiple outputs parallelly based on the data (document) provided by the user. For ex, User want to generate notes and the quiz based on the document now what we gonna do, Pass the document to Model 1 for notes, In parallel, send it to Model 2 for quiz generation, Finally, combine outputs using Model 3 to merge the results into a single output.

`RunnableParallel`: With this parallelly we can execute multiple chains.

`4. Conditional Chain`: Executes tasks based on specific conditions means sentiment wise, for ex, for making a feedback generator let's say want to analyze customer sentiment let if positive then praise him but if for negative then say sorry / send email to the customer care team. Now most important we might get result sometimes not perfect like so we use pydantic for proper classify so always we get in a proper way not gamble or mixing because pydantic gives us the proper validation and thus parsering gives us perfect output always.

**RnnableBranch: It runs on the conditional way like for ex, if first_condition true then gives first chain same for second otherwise the default chain seems like if, elif, else.**

---

# Memory: 
- Memory is the key component while making AI agents and these can actually save every single chat history, so you able to interact any prevoius one you don't need to tell the LLM from scratch, LLM can remember from your chat history and you'll get accurate answers.

Example:

- Just like the chatgpt you can see the history of every chat and when you ask to generate a report based on your all of the chats so gpt gives you whole history.

## Why LangChain Changed the Memory System

- *Old memory system works fine (`ConversationBufferMemory`) for one user and his chat history even stored in plain text but had limits like it doesn't provide multi-role support (system, huma, AI), hard to integrate in other runnable systems and not good for mulitple users*
- *But sooner langchain integrate (`RunnableWithMessageHistory`) in memory just like with `PromoptTemplate(), invoke(), retriever()`*
- *Instead of storing plain text it store multi-role format for better readability and each user store in sessions with their id's (like user_1, user_2 .....n) so multiple users are allowed and manage their chat histories easily.*


---
# Runnables

- It is unit of work means you give input -> process -> then we get output. It provides common interface like invoke() where you give input and get output, similarly batch() handles multiple inputs and able to give multiple answers, also stream() method extract stream output from input. So as all runnables have common interface so you can connect each runnable with other and able to execute complex workflow even you connect multiple runnable and able to interact easily.

## Why Runnable

**Before runnables, LangChain had different modules for different tasks (LLM calls, retrievers, tools, etc.). Each module behaved differently.**

- Developers had to learn each one separately.

- Combining them was frustrating and slowed down projects.

- Workflows became messy and hard to maintain.

👉 The LangChain community realized they needed a standardized interface for everything.

So they introduced runnables:

- A common format for all components.

- Easy to connect, mix, and extend.

- Makes it possible to build workflows that are sequential, parallel, or conditional.

This change is a big reason why LangChain became so popular for GenAI apps.

If want to understand how runnable work? Then think about "Lego blocks"

---

## Types of runnables:

**1. Task Specific Runnables**

-   These are the core components (LLMs, prompts, retrievers) wrapped into runnables.
They perform actual AI tasks.

Examples:

1. `ChatOpenAI()`: (so it runs llm model) It's components and task specifc llm model.
2. `PromptTemplate()`: It make prompts dynamically.
3. `Retriever()`: Fetch related documents.

- Use when u want to build text generation, prompt dynamically and retrive docs.

**2. Runnable Primitives: They are helper funcs to connect & interact with tasks specific runnables They don't do AI tasks themselves, instead they define how runnables can interact.**

- Use case: They help to connect & make interactions with different runnables to interact (sequentially, parallel, conditionally etc).


1. RunnableSequence: It helps to connect different runnables and form a chain for ex, if we have 2 runnables like R1 and R2 then with this we can connect them and output of R2 is input of R1 which really help full think about "deep search" in chatgpt. How each prompt connect with another and form a chain which forms multi LLM's for long & sequncial answers.

- Example:
```
Question -> PromptTemplate -> ChatOpenAI -> Output
```

2. RunnableParallel: Using for connect two different LLM's for specific topic for ex, Topic = AI -> LLM-1 -> tweet -> generate tweet -> getting output, now with same topic AI -> LLM-2 -> generate LinkedIn post -> getting output. Remember both runnables work parallelly means independently executed but with same input (topic).

- Example:
```
Runnable_1 = Topic -> LLM_1 -> Generate Tweet
Runnable_2 = Topic -> LLM_2 -> Generate Linkedin Post
```

3. RunnablePassthrough: It's return the whole input as it is in output even without modification and using for getting original input along output and it's sub-output.

- Example:
```
If I want to get a joke and based on the joke I want explanation along with the joke so there I can use it.
```

4. RunnableLambda: You can able to convert python functions into runnable and able to give your code to LLM's workflows.

- Example:
``` 
- Topic -> prompt -> LLM ->  generate output -> generate number of words.
```

5. RunnableBranch: Uses for conditional chain like (if else).

- Example:
```
- email -> LLM (For classify the mood / behaviour of the mail) -> put it into branches (positive, question, negative, refund) -> do according that.
```

## Q. Why we use “|” rather than “RunnableSequence”?

- **`RunnableSequence` is class that chains runnable, but langchain introduced (LCEL) LangChain Expression Language is shortcut to connect runnable (chain) just like runnable sequence class for easier setup and provides simplicity**
---

# RAG

**RAG is a technique that combines information retrieval with language generation where model retrieves relevant documents from a knowledge base and then uses them as context to generate accurate and grounded responses.**


## Benefits of using RAG,

- 1. Use of up-to-date information.

- 2. Better privacy. -> If u want to extract document in a human friendly language from the document and give to the chatgpt then it blocks due to contextual limitations even security of your personal / private data leaks. So that's why u use RAG based app for data privacy.

- 3. No limit of document size. -> There is no limitation of the data you're providing to the RAG agent but need good hardware for faster responses.

## RAG Componenets:

- 1. Document loaders: These are components in langchain used to load data from various sources. For ex. pdf, txt, db or cloud etc. So we need standardize format before we send it to the next components like splitters, vector, stores and retrievers. Document format will be like,

Document(
   page_content = "The actual document page content.",
   metadata = {"source": "filename.pdf", .....}
)

## Types of Document Loaders

- 1.1 TextLoaders: It can be use only use .txt file to read plain text and converts them into the langchain document objects.

- 1.2 PyPDFLoader: It can be used to convert pdf files into each page document object. Limitation is only read textual content from pdf not because use "PyPDF" module. So there are another modules like if you have pdf's with table/columns then use "PDFLumberLoader", same as want best structure and scanned/image pdfs then "UnStructurePDFLoader" or "AmazonTextractPDFLoader", need layout and image data then "PymuPDFLoader".

- 1.3 DirectoryLoader: You can use this when you want to load multiple documents from a directory.

```python
1. "**/*.txt": All .txt files in subfolders.
2. "*.pdf": All .pdf files from root directory.
3. "data/*.csv": All .csv files from data folder.
4. "**/*": All type of files any type of folders.
```

- 1.4 WebBasedLoader: You can load web page contents in your langchain and then use like Q & A. So internally it uses request module for requesting URL's then beautifulsoup for HTML and extract text. We can use to extract blogs, news articles or public websites into a text readable format and static. So it doesn't work java-script heavy pages instead you can use SeleniumURLLoader for that.

- 1.5 CSVLoader: To load csv files into langchain.

2. text splitters
3. Vector stores(VectorDB)
3. Retrievers

---

# Text Splitters

- Text splitters can be very essential part in langchain for dividing large data like (like articles, blogs, html pages, large pdfs etc..) into smaller chunks so llm can handle them effectively.

### Why text splliters are important?

1. Overcome model limitations: LLM's have limited storage for handling data so text spliters help to overcome this problem.

2. Downstream tasks: Text splitting imrproves nearly every LLM poweered tasks like, embeddings (short chunks form in numbers can help system to find more accurate vectors), semantic search (search result point to focused info) and proper summerization (prevents hallutinations).

### Types of text splitting

1. Length based text splliting: Means you already give the length threshold for dividing chunks at how much length, like char based splliter. Only disadvantage is to strictly follow-up on threshold occur lingustic and grammatical problem on chars means stopped st desired threshold doesn't see if the char is complete or not thus semantic search is partially not recommended.

2. Text Structure Based: Also known for (RecursiveTextSplitter) and one of te most use full technique. So it inheritely follow structure means organize your text in way like, first organize into paragraphs -> sentences -> words.

- For paragraph we have = `"\n\n"`
- similarly for line changes = `"\n"`
- For spaces "_" and for character = `" "`.

3. Document Based Text Splitter: Here exactly the same as the previous (RecursiveTextSplitter) but the difference is only you can use this for different kind of document formats like html, python, java-script etc..

4. Semantic Meaning Based: They split the text on the meaning based rather than length or structured based. We basically used embedding models to get the similarity score then through we get the idea if the score is low then topic is change and then it perform splitting, so basically it uses user window sliding technique. Currently this is on the experimental based in future we can use this for making strong ai agents.

---

# Vector Store

- Vector store is a system design store and retrieve the data represented as numerical vector, so basically when we generate embeddings we need to store somewhere but rather than using traditional db's like MySQL we can use vector store.

### Why vector store?

- Retrieve similarity score thru semantic search from whole context is far better than matching keyword.

### Features of Vector Store:

1. Storage: You can store in your memory for quick lookups or on disk for durability for large scale use.

2. Similarity Search: Helps to retrieve the highest similarity score vector.

3. Indexing: Able to find nearest neighbour like provide data-structure that enables fast similarity search on high dimensional vectors.

4. CRUD Operation: Performing on vectors like adding, reading, updating, removing outdated vectors.

- Use-cases:

1. Semantic Search
2. RAG
3. Recommender Systems
4. Image/Multimedia Search

### Difference between Vector Store and Vector Database

- Mostly same, difference is only that if you need storage and retriever option so that system called vector store, otherwise if you get database kind of features like distributed (means scale far you want), backup feature (in case of losing data), ACID transaction, Concurrency control (multiple user can interact), authentication to avoid third party add.

- In simple terms vector store is a light weight library that can focuses on storing vectors (embeddings) and performs semantic search.

- In simple terms vector database can store vectors with extra database features (clustering, security, metadata filtering and durability).

- Example of vector store: FAISS (A Facebook library)
- Example of vector database: Milvus, Qdrant, Weaviate, Pinecone.


### Vector stores in LangChain

1. Chroma Vector Store

- Chroma is a light weight, open-source vector database that is especially friendly for local development and small-to-medium-scale production needs.

- Supported Stores: LangChain integrates with multiple vector stores (FAISS, Pinecone, Chroma, Qdrant, Weaviate, etc.), gives you flexibility in scale, features and deployment.

- Common Interface: Easy to switch between on vector store to another (ex FAISS -> Pinecone (for data extend)).

- metadata handling: Provides the source means just like chatgpt gives you source links while web search.

---

# Retrievers

Retrievers: In langchain retrievers thar fetches the doc from data source in response to the user query. Remember each retriever are runnables.

There are several types of retrieves.

It depends if whether it depends on which data source is using like Wikipedia, vector store, arxiv store and what search strategy is using like maximal margin, multi-query or contextual comparison.

1. Wikipedia Retriever: Basically it hit the Wikipedia API and fetch the relevant content as per given query.

Example:

You given a query ("Where is Paris?")
It sends the wiki's API
Then fetch the relevant content
return langchain document objects.

2. Vector Store Retriever: It's common retriever we use while makin RAG based agents where it fetches document from vector store based on semantic search on vector.

Example:

U store doc in vector store (FAISS, Chroma, Weaviete).
each dense vector turned into dense vector an embedding vector
so the retrieve compare the user query vector into vector store basically performs semantic search
return relevant top k frequent result.

MMR: how can we pick result which are not just similar but relevant to the other also differ from each other?

3. MMR is an informational retravel algorithm design to reduce redundancy but still getting high relevance result. So it's not repeating but still u get high relevant result. So it avoid pick most relevant result first and then find least similar result.

So it helps RAG pipelines with when you want full contextual window to contain diverse but still relevant information. Especially useful when documents semantically overlapping.

4. Multi-Query Retriever: It helps to solve ambiguous user's query with generating multi-query and send to the LLM then retrieve top k results.

Example:

User: How can I stay healthy? (very broad question right?)

So this have several meaning like what should I eat, how often should I exercise or how can I manage stress. Now with multi-query it generates multiple questions based on the original query then each of them find in document and merge in last show top k result.

5. Contextual Compression Retriever: It improves retrieval quality where returns only relevant info rather than entire paragraph in old ways as per your query you get relevant info along with whole paragraph in short getting output but returns as it is not compressed that's what contextual compression retriever do it returns only one which is relevant to the query.

-> Other Retrievers

---

# Problems with current LLM's and fine-tuning

- Current problems with LLM's

1. Private Data
2. Halucination
3. Past trained (which can not conclude current info)

- The answer to overcome this problem is `Fine-Tuning`.

- Where you give your domain specific data to the pre-training

- There are several ways to perform fine-tuning like "supervised training" where you provide labeled data like prompt -> desired output second technique is "continued pretraining" like unsupervised training where you give whole transcript to the LLM just how they naturally made and third technique is "RLHF" where you perform based on reinforcement learning (like human-feedback training) to learn what to do in real-world scenarios. Even there are other techniques like Lora, QLora etc..

### The problem with "Fine-Tunning"

- Computationally expensive when fin-tuning a whole LLM
- Need strong expertise means experienced AI engineers, data-scientist team for train fin-tuning model.
- Fine-Tuning is not useful for training on where new data comes very fast.

-- So eventually we solve these problems to use such different techniques like,

1. In-Context learning: These technique work exactly we use latest LLM's in daily life like GPT-3/4, gemini, claude or llama etc in way that you describe a problem along with approach (example) for solving a problem to LLM so based on these LLM can able to solve your problem by given your example. this learning is also called emergent property.

- So RAG is a way to make a language model (like chatgpt) with giving extra information along with your question.
- So the flow is like query & context -> prompt -> LLM -> Response.

***Total 4 steps to make RAG based systems.***

1. Indexing: Is useful when you connect external based knowledge thru text loaders, text splitters, generate (vectors)embeddings and perform semantic search, store vectors in vectordb, retriever.
2. Retrieval: Retrieve only relevant chunks (information whose higher similarity score) rather than whole thing.
3. Augmentation: Prompt generation based on the question and context which is called augmentation.
4. Generation: So LLM generates the response based on the prompt which is called generation.

- So at last for fine-tuning RAG is the cheaper solution because you do with indexing rather than perform whole re-training your model.

---

# LangChain Tools

LangChain Tools: Assume current LLM's as a human body and mind able to think and provide approaches to how to perform but not executable learning (for example book tickets, access live data, run code, interact with DB, calls API's Do reliable math etc.) like hand and legs so that tools can do it.

There are two types of tools, 

1. Built-In tools: built-in tools (made by community), production ready and minimal or no setup needed use with directly importing.

1.1 DuckDuckGoSearchRun: Web serach via DuckDuckGo
1.2 WikipediaQueryRun: Wikipedia summary
1.3 PythonREPLTool: Run raw Python code
1.4 ShellTool: Run shell commands
1.5 RequestsGetTool: Make HTTP GET requests
1.6 GmailSendMessageTool: Send emails via Gmail
1.7 SlackSendMessageTool: Post message to Slack
1.8 SQLDatabaseQuearyTool: Run DQL queries

and many more

2. Custom Tools: Made for special usecases or personal usecases as per the requirement.

Why custom tool needed means what type of scenarioes?

-> If you want interact with your own API's, implement your business specific logic, continues itractions with your databases, product or app.

-> There are many ways to create a custom tools but amongst them using "@ (decorator)", strucutred & pydantic and BaseTool class are popular.

How tools fit into the AI agent ecosystem?

-> AI agents an LLM's powered system that can be think, make decision and action taking capabilities using external tools or API's achieve a goal.

Flow: AI agent made by two things 1. LLM for reasoning and decision making and 2. Tools for action.

Toolkits

You can eventually combine your related tools to form a toolkit like "GoogleDriveToolKit" combines
"GoogleDriveCreateFileTool" for upload a file, "GoogleDriveSearchTool" for earch for a file by name/content, "GoogleDriveReadFileTool for read contents of a file and so on.

---

# LangChain Tool Calling


Tool Binding: Tool binding is the process to connect your LLM with tools and perform input ready tasks, so that way LLM knows,

1. What tools are available (name)
2. It knows each tools via description to tell LLM what tasks your tool perform
3. What input format to use via schema means simply you're providing input template (like "PromptTemplate").

Tool Calling: is the process where your LLM decides, during conversation to execute a specific tool when needed and generate structured output.

Note: LLM doesn't execute the tool it suggest you tools otherwise execution can be done via LangChain or you, it depends on your logic.

Tool Execution: Tool execution is the step where the actual python (tool) is run using the input arguments as suggest tool by the LLM.

Example:

The LLM says:

"Hey, call the "multiply" tool with a=8 and b=7

Langchain or you (programmer) is actually run:

multiply(a=8, b=7)
-> and get the result: 56

---

# AI-Agents

ReAct: ReAct is a design pattern used in AI agents can do Reasoning + Acting. It allows a LLM to perform internal reasoning (thought) with external actions (like tool use) in a structured, multi-step process. In simple meaning instead of generating an answer in one go, the model thinks step by step, deciding what to do and what needs and optionally call tools (APIs, calculators, web search, etc.) to help it.

ReAct is useful for,

- Multi-step problem
- Tool-augmented tasks (web search, database lookup, etc.)

Agent Executor: It is responsible to run Thought, Action and Observe loop continuously.

Steps how agent executor do in back-end.

1. Executor send user query along with thought trace (means monitor the sequence).
2. Agent think and plan which tool can be useful, so in-response agent send action and action input to the executor.
3. Now Executor go through that tool perform action as per the query and send to agent.
4. Now executer send the user query with updated though trace (along with output) to the agent.
5. Lops start again until gets the final answer means starting from first step.

Creating Agent

- Basically it requires chose react / any agent method, LLM, tools (means which tool is appropriate for performing action) and prompt.
