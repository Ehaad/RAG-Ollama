import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
import time


st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #fdf6e3;  
        color: #2c2c2c; 
        font-family: "Courier New", monospace;
    }
    h1, h2, h3 {
        font-family: "Courier New", monospace;
        color: #222222;
        text-shadow: 1px 1px #e0e0e0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f4e1c1;
        color: #000000;
    }
    section[data-testid="stSidebar"] h1, h2, h3, label, span {
        color: #000000 !important;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        background: #d6b58c;
        color: #000;
    }

    /* Input box */
    .stTextInput>div>div>input {
        background: #fffdf5;
        color: #000000;
        border: 2px solid #d4a373;
        border-radius: 6px;
        padding: 10px;
        font-family: "Courier New", monospace;
        font-size: 15px;
    }

    /* Buttons */
    .stButton>button {
        background: #e0c097;
        color: #000;
        border: 2px solid #a47148;
        border-radius: 10px;
        font-family: "Courier New", monospace;
        font-size: 15px;
        padding: 6px 20px;
        box-shadow: 3px 3px 0px #7a4e2d;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #d6b58c;
        color: #000;
        transform: translate(-2px, -2px);
        box-shadow: 5px 5px 0px #5c3b24;
    }

    /* Answer box */
    .answer-box {
        background: #fffaf0;
        border: 2px solid #d9c2a3;
        padding: 16px;
        margin-top: 15px;
        font-family: "Courier New", monospace;
        font-size: 17px;
        color: #000000;
        box-shadow: 0px 0px 10px #e6d5b8;
        border-radius: 8px;
        line-height: 1.6;
    }

    /* Context documents */
    .doc-box {
        background: #fff7e6;
        border: 2px dashed #d4a373;
        padding: 12px;
        margin-top: 10px;
        font-family: "Courier New", monospace;
        font-size: 14px;
        color: #2c2c2c;
        border-radius: 6px;
        max-height: 180px;
        overflow-y: auto;
    }

    /* Inline code */
    code {
      color: #c7254e !important;
      background-color: #f9f2f4 !important;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: monospace !important;
    }

    /* Block code */
    pre, code {
        color: #2c2c2c !important;
        background-color: #f5f5f5 !important;
        font-family: "Courier New", monospace;
        font-size: 14px;
        padding: 10px;
        border-radius: 6px;
        display: block;
        overflow-x: auto;
    }

    /* Streamlit rendered code blocks */
    .stCodeBlock pre {
        background-color: #f5f5f5 !important;
        color: #2c2c2c !important;
        font-family: monospace !important;
        font-size: 14px !important;
        padding: 10px;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


model_name = st.sidebar.selectbox(
    "Choose AI Model:",
    ["llama3.2:3b","llama3","Groq","Gemma3n","Mistral"]
)

if "vectors" not in st.session_state or st.session_state.get("model_name") != model_name:
    st.session_state.embeddings = OllamaEmbeddings(model=model_name)
    st.session_state.loader = WebBaseLoader("https://huggingface.co/docs/transformers/en/training#fine-tuning")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs
    )
    st.session_state.vectors = Chroma.from_documents(
        st.session_state.final_documents, st.session_state.embeddings
    )
    st.session_state.model_name = model_name

st.markdown("<h1 align='center'> RAG Pipeline <br> Intelligent Document Search with Ollama </h1>", unsafe_allow_html=True)

llm = ChatOllama(model=model_name)

prompt_template = ChatPromptTemplate.from_template(
    """Answer the questions based on the provided context only.
<context>
{context}
</context>
Question: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

user_prompt = st.text_input("💬 Enter your query:")

if st.button("Run Query"):
    if user_prompt:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        st.markdown(
            f"<div class='answer-box'>✅ Answer:<br>{response['answer']}</div>",
            unsafe_allow_html=True
        )
        st.write("⚡ Processing time:", round(time.process_time() - start, 2), "seconds")

        with st.expander("📂 Context documents"):
            for i, doc in enumerate(response["context"]):
                st.markdown(
                    f"<div class='doc-box'>[Doc {i+1}] {doc.page_content}</div>",
                    unsafe_allow_html=True
                )
