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
```
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
{"Time": "Morning", "Activity": "Enjoy dinner at a Seine riverside cafe"}
]
```

## 🔹Q. Now why would we need structured output?

- Ans: Generally structured output will return well-defined data format (example: JSON) rather than free-form text. Now this makes really useful to work or integrate LLM's with other systems too.

- **Data Extraction:** For storing a data into database from key data extraction from data then we make structured data like in JSON then store in DB.

- **API Building:** For building API's from structured data soa anyone can use across the world.

- **Agents:** AI agents need structured data for accurate actions.

## 🔹How many types of LLM's can generate structured output:

- Some LLMs (e.g., OpenAI, Google, Anthropic) provide structured output options:

1. `with_structured_output()` → Built-in function to enforce structured responses

2. `output_parser()` → For models that don’t support strict structure natively

- `with_structured_output` there is a param named "method" has provide two ways to generate structured output, 1. 'JSON mode' (while working with claude or jemini because they only support json format) 2. 'function calling' (while working with openai for calling function (agents)).

- Data format before call structured output function we need format our data in three ways we can make,

## 1. TypedDict: Nothing but your data format would be in keys and values where we just need to specify what to put in keys and values. It helps where we define that what type of datatype needed with key int, str etc.. It prevents runtime error while new one programmer come and change the type but there is cons like it can't perform validation like suppose if you provide int and the value you provide value in int in str format it still gives you in str format means it doesn't care about the validation.

## 2. Pydantic (mostly uses in FastAPI): Data validation & parsing library and ensures the data you work with is correct, structured and safe-search. 

## 3. json_schema: You can use where your project made with multiple languages like backend made in python and frontend in java-script where both side you can't use either TypedDict or Pydantic you need common format where JSON format is universally accepts any language so that purpose we use JSON schema.
