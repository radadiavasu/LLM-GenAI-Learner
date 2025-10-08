import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
import os
import tempfile
from typing import List, Dict
import time
import requests
from bs4 import BeautifulSoup
import json

# Page configuration
st.set_page_config(
    page_title="📚 AI Learning Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        background-color: #121110;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .source-box {
        background-color: #e8eaf6;
        border-left: 4px solid #3f51b5;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .web-source-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .comparison-box {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-msg {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .deep-search-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'current_pdf_name' not in st.session_state:
    st.session_state.current_pdf_name = None
if 'deep_search_mode' not in st.session_state:
    st.session_state.deep_search_mode = False
if 'web_top_k' not in st.session_state:
    st.session_state.web_top_k = 5

@st.cache_resource
def initialize_llm():
    """Initialize the LLM and chat interface"""
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-70B-Instruct",
        task="conversational",
        temperature=0.7,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=llm)

@st.cache_resource
def get_embeddings():
    """Initialize embeddings model"""
    return SentenceTransformerEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={"trust_remote_code": True},
    )

@st.cache_resource
def initialize_web_search():
    """Initialize web search tool"""
    search = DuckDuckGoSearchAPIWrapper(max_results=5)
    return DuckDuckGoSearchResults(api_wrapper=search)

def process_pdf(uploaded_file) -> tuple:
    """
    Process uploaded PDF and create vector store
    Returns: (vectorstore, document_count, error_message)
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        doc_splits = text_splitter.split_documents(docs)
        
        # Create embeddings and vector store
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(doc_splits, embeddings)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return vectorstore, len(doc_splits), None
        
    except Exception as e:
        return None, 0, str(e)

def perform_web_search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Perform web search and return top k results
    Returns: List of dicts with 'title', 'snippet', 'link'
    """
    try:
        search_tool = initialize_web_search()
        results = search_tool.run(query)
        
        # Parse results
        web_results = []
        try:
            # DuckDuckGo returns results as string, parse it
            results_list = eval(results) if isinstance(results, str) else results
            for i, result in enumerate(results_list[:top_k]):
                web_results.append({
                    'number': i + 1,
                    'title': result.get('title', 'No title'),
                    'snippet': result.get('snippet', 'No description available'),
                    'link': result.get('link', '#')
                })
        except:
            # Fallback parsing if eval fails
            st.warning("Web search results format unexpected, using alternative parsing")
        
        return web_results
        
    except Exception as e:
        st.error(f"Web search error: {str(e)}")
        return []

def get_answer_with_sources(question: str, vectorstore, llm) -> Dict:
    """
    Get answer with source citations (regular mode)
    Returns: dict with 'answer', 'sources', and 'source_documents'
    """
    try:
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 4}
        )
        
        # Get relevant documents
        relevant_docs = retriever.invoke(question)
        
        # Create context from documents
        context = "\n\n".join([
            f"[Source {i+1}]:\n{doc.page_content}" 
            for i, doc in enumerate(relevant_docs)
        ])
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful educational assistant. Your role is to:
1. Explain concepts in simple, easy-to-understand language
2. Break down complex topics into digestible parts
3. Use examples and analogies when helpful
4. Keep explanations concise but comprehensive
5. Always mention which source ([Source 1], [Source 2], etc.) supports your statements

Be friendly, encouraging, and make learning enjoyable!"""),
            ("user", """Question: {question}

Context from the document:
{context}

Please provide a clear, simple explanation based on the context above. Mention the source numbers when you reference information.""")
        ])
        
        # Get answer
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "question": question,
            "context": context
        })
        
        # Prepare sources
        sources = []
        for i, doc in enumerate(relevant_docs):
            source_info = {
                'number': i + 1,
                'content': doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                'page': doc.metadata.get('page', 'N/A')
            }
            sources.append(source_info)
        
        return {
            'answer': answer,
            'sources': sources,
            'source_documents': relevant_docs,
            'mode': 'regular'
        }
        
    except Exception as e:
        return {
            'answer': f"Sorry, I encountered an error: {str(e)}",
            'sources': [],
            'source_documents': [],
            'mode': 'regular'
        }

def get_deep_search_answer(question: str, vectorstore, llm, top_k: int = 5) -> Dict:
    """
    Get answer combining web search and document retrieval
    Returns: dict with 'answer', 'doc_sources', 'web_sources', 'comparison'
    """
    try:
        # Step 1: Perform web search
        web_results = perform_web_search(question, top_k)
        
        # Step 2: Get document sources
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 4}
        )
        relevant_docs = retriever.invoke(question)
        
        # Step 3: Prepare contexts
        doc_context = "\n\n".join([
            f"[Document Source {i+1}]:\n{doc.page_content}" 
            for i, doc in enumerate(relevant_docs)
        ])
        
        web_context = "\n\n".join([
            f"[Web Source {result['number']}]:\nTitle: {result['title']}\nContent: {result['snippet']}\nURL: {result['link']}" 
            for result in web_results
        ])
        
        # Step 4: Create comprehensive prompt with comparison
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an advanced educational assistant with access to both document content and real-time web information. Your role is to:
1. Provide comprehensive answers using BOTH document and web sources
2. Compare information from the document with current web information
3. Highlight any similarities, differences, or updates
4. Explain concepts clearly with examples
5. Cite sources appropriately ([Document Source X] or [Web Source Y])
6. Note if information is more current online or if document provides deeper detail

Be analytical, thorough, and help users understand the complete picture!"""),
            ("user", """Question: {question}

DOCUMENT CONTEXT:
{doc_context}

WEB SEARCH RESULTS:
{web_context}

Please provide:
1. A comprehensive answer using both sources
2. Comparison between document and web information
3. Note any updates, contradictions, or complementary information
4. Cite sources clearly""")
        ])
        
        # Get answer
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "question": question,
            "doc_context": doc_context,
            "web_context": web_context
        })
        
        # Prepare document sources
        doc_sources = []
        for i, doc in enumerate(relevant_docs):
            source_info = {
                'number': i + 1,
                'content': doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                'page': doc.metadata.get('page', 'N/A')
            }
            doc_sources.append(source_info)
        
        return {
            'answer': answer,
            'doc_sources': doc_sources,
            'web_sources': web_results,
            'mode': 'deep_search'
        }
        
    except Exception as e:
        return {
            'answer': f"Sorry, I encountered an error in deep search: {str(e)}",
            'doc_sources': [],
            'web_sources': [],
            'mode': 'deep_search'
        }

def display_sources(sources: List[Dict]):
    """Display document source citations"""
    if sources:
        st.markdown("### 📄 Document Sources")
        for source in sources:
            with st.expander(f"📄 Document Source {source['number']} (Page {source['page']})"):
                st.markdown(f"```\n{source['content']}\n```")

def display_web_sources(web_sources: List[Dict]):
    """Display web search results"""
    if web_sources:
        st.markdown("### 🌐 Web Sources")
        for source in web_sources:
            with st.expander(f"🔗 Web Source {source['number']}: {source['title']}"):
                st.markdown(f"**Summary:** {source['snippet']}")
                st.markdown(f"**Link:** [{source['link']}]({source['link']})")

# Main App Layout
st.title("📚 AI Learning Tutor")
st.markdown("*Learn = Earn, Learn smarter not harder.*")

# Sidebar for PDF upload and settings
with st.sidebar:
    st.header("📁 Upload Your Study Material")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload the PDF you want to learn from"
    )
    
    if uploaded_file:
        if st.button("🔄 Process PDF", type="primary"):
            with st.spinner("📖 Reading and understanding your document..."):
                vectorstore, chunk_count, error = process_pdf(uploaded_file)
                
                if error:
                    st.error(f"❌ Error processing PDF: {error}")
                else:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_processed = True
                    st.session_state.current_pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                    
                    st.markdown(f"""
                    <div class="success-msg">
                        ✅ Successfully processed!<br>
                        📄 <b>{uploaded_file.name}</b><br>
                        📊 {chunk_count} chunks created
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(1)
                    st.rerun()
    
    if st.session_state.pdf_processed:
        st.success(f"✅ Ready: {st.session_state.current_pdf_name}")
        
        if st.button("🗑️ Clear & Upload New PDF"):
            st.session_state.vectorstore = None
            st.session_state.pdf_processed = False
            st.session_state.chat_history = []
            st.session_state.current_pdf_name = None
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 Tips for Better Learning")
    st.markdown("""
    - Ask specific questions about topics
    - Use Deep Search for current information
    - Compare document vs web insights
    - Request examples or analogies
    - Ask for summaries of complex topics
    """)

# Main chat interface
if st.session_state.pdf_processed and st.session_state.vectorstore:
    
    # Display chat history
    for chat in st.session_state.chat_history:
        # Show mode badge
        if chat.get('mode') == 'deep_search':
            st.markdown('<div class="deep-search-badge">🔍 Deep Search Mode</div>', unsafe_allow_html=True)
        
        with st.chat_message("user"):
            st.write(chat['question'])
        
        with st.chat_message("assistant"):
            st.write(chat['answer'])
            
            if chat.get('mode') == 'deep_search':
                # Show both document and web sources
                col1, col2 = st.columns(2)
                
                with col1:
                    if chat.get('doc_sources'):
                        with st.expander("📚 View Document Sources"):
                            display_sources(chat['doc_sources'])
                
                with col2:
                    if chat.get('web_sources'):
                        with st.expander("🌐 View Web Sources"):
                            display_web_sources(chat['web_sources'])
            else:
                # Regular mode - show only document sources
                if chat.get('sources'):
                    with st.expander("📚 View Sources"):
                        display_sources(chat['sources'])
    
    # Mode selector with popover (3 dots) at bottom left
    col_mode, col_input = st.columns([0.2, 6])
    
    with col_mode:
        # Get current mode display
        current_mode_icon = "🔍" if st.session_state.deep_search_mode else "📄"
        current_mode_text = "Deep Search" if st.session_state.deep_search_mode else "Research"
        
        # Create popover with 3-dot menu
        with st.popover("⋮", help=f"Current mode: {current_mode_text}"):
            st.markdown("**Select Search Mode**")
            
            if st.button(f"📄 Research Mode {'✓' if not st.session_state.deep_search_mode else ''}", 
                        key="mode_regular", 
                        use_container_width=True,
                        type="primary" if not st.session_state.deep_search_mode else "secondary"):
                st.session_state.deep_search_mode = False
                st.rerun()
            
            # st.markdown("<small>Search only in document</small>", unsafe_allow_html=True)
            
            # st.markdown("---")
            
            if st.button(f"🔍 Deep Search Mode {'✓' if st.session_state.deep_search_mode else ''}", 
                        key="mode_deep", 
                        use_container_width=True,
                        type="primary" if st.session_state.deep_search_mode else "secondary"):
                st.session_state.deep_search_mode = True
                st.rerun()
            
            st.markdown("<small>Search in document + web</small>", unsafe_allow_html=True)
    
    with col_input:
        # Chat input
        user_question = st.chat_input(f"{current_mode_icon} Ask anything...")
    
    if user_question:
        # Determine if deep search is enabled
        deep_search_enabled = st.session_state.deep_search_mode
        
        # Add user message to chat
        with st.chat_message("user"):
            st.write(user_question)
        
        # Get answer based on mode
        with st.chat_message("assistant"):
            llm = initialize_llm()
            
            if deep_search_enabled:
                # Deep Search Mode
                with st.spinner("🔍 Performing deep search (document + web)..."):
                    result = get_deep_search_answer(
                        user_question,
                        st.session_state.vectorstore,
                        llm,
                        st.session_state.web_top_k
                    )
                    
                    st.write(result['answer'])
                    
                    # Show sources in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if result['doc_sources']:
                            with st.expander("📚 View Document Sources"):
                                display_sources(result['doc_sources'])
                    
                    with col2:
                        if result['web_sources']:
                            with st.expander("🌐 View Web Sources"):
                                display_web_sources(result['web_sources'])
                
                # Save to chat history
                st.session_state.chat_history.append({
                    'question': user_question,
                    'answer': result['answer'],
                    'doc_sources': result['doc_sources'],
                    'web_sources': result['web_sources'],
                    'mode': 'deep_search'
                })
            else:
                # Regular Mode
                with st.spinner("🤔 Thinking..."):
                    result = get_answer_with_sources(
                        user_question,
                        st.session_state.vectorstore,
                        llm
                    )
                    
                    st.write(result['answer'])
                    
                    if result['sources']:
                        with st.expander("📚 View Sources"):
                            display_sources(result['sources'])
                
                # Save to chat history
                st.session_state.chat_history.append({
                    'question': user_question,
                    'answer': result['answer'],
                    'sources': result['sources'],
                    'mode': 'regular'
                })
        
        st.rerun()
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Welcome screen
    st.markdown("""
    ## 👋 Welcome to Your AI Learning Assistant with Deep Search!
    
    ### 🚀 Two Powerful Modes:
    
    #### 📄 Regular Mode
    - Get answers directly from your PDF document
    - Fast and focused on your study material
    - Perfect for exam preparation and quick reviews
    
    #### 🔍 Deep Search Mode (NEW!)
    - Combines document analysis with real-time web search
    - Compare information from your PDF with current web sources
    - Get comprehensive insights with multiple perspectives
    - Identify updates, contradictions, or complementary information
    - Best for research and staying current
    
    ### Features:
    - 🎯 **Smart Analysis** - AI understands context and nuance
    - 📚 **Source Citations** - See exactly where info comes from
    - 🌐 **Web Integration** - Access latest information online
    - 💡 **Comparative Insights** - Understand different viewpoints
    - 🔄 **Interactive Learning** - Ask follow-up questions naturally
    
    ### How to Get Started:
    1. 📤 Upload a PDF document using the sidebar
    2. 🔄 Click "Process PDF" to analyze it
    3. 🔍 Choose your search mode (Regular or Deep Search)
    4. 💬 Start asking questions
    5. 📖 Review sources and comparisons
    
    ### Example Questions:
    **Regular Mode:**
    - "Explain [concept] from the document"
    - "Summarize chapter 3"
    - "What are the key points about [topic]?"
    
    **Deep Search Mode:**
    - "How does this document's view on [topic] compare with current research?"
    - "Are there any updates on [concept] since this was written?"
    - "What do recent sources say about [subject]?"
    """)
    
    st.info("👈 Start by uploading a PDF from the sidebar!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    Made with ❤️ for better learning | Powered by LangChain, HuggingFace & DuckDuckGo
</div>
""", unsafe_allow_html=True)