import tempfile
import streamlit as st
import uuid

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pinecone

# -------------------
# Session Setup
# -------------------
# Assign a unique session_id for each user (persists during their session)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# -------------------
# API Keys & Config
# -------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]  # e.g., "us-east-1-aws"
INDEX_NAME = "rag-index"

# -------------------
# Initialize Pinecone (v3 client)
# -------------------
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create index if it doesn’t exist
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=1536,  # must match OpenAI embedding size
        metric="cosine"
    )

# -------------------
# Embeddings & Vector Store
# -------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,
)

vector_store = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

# Each retriever is scoped to the current session only
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5, "filter": {"user": st.session_state.session_id}}
)

# -------------------
# LLM
# -------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Lecture Notes Assistant", layout="wide")
st.title("📚 Lecture Notes Q&A Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------
# Upload PDF Section
# -------------------
uploaded_files = st.file_uploader(
    "📂 Upload up to 3 lecture notes (PDFs)",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    if len(uploaded_files) > 3:
        st.warning("⚠️ You can only upload a maximum of 3 PDFs per session.")
        uploaded_files = uploaded_files[:3]

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # Try PyPDFLoader first, fallback to PyMuPDFLoader
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
        except Exception:
            loader = PyMuPDFLoader(pdf_path)
            pages = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(pages)

        # Add metadata: source file + user session
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["user"] = st.session_state.session_id

        # Add documents to Pinecone
        vector_store.add_documents(documents)
        st.success(
            f"✅ Added {len(documents)} chunks from {uploaded_file.name} to Pinecone DB (private to your session)"
        )

# -------------------
# Chat Interface
# -------------------
st.subheader("💬 Chat with your notes")

user_question = st.chat_input("Ask a question...")

if user_question:
    st.session_state.chat_history.append(("user", user_question))

    # Retrieve relevant chunks
    relevant_notes = retriever.invoke(user_question)
    context = "\n\n".join([doc.page_content for doc in relevant_notes])

    # Build prompt
    prompt = f"""
    You are an expert in this subject.
    Use the following lecture notes to answer the student's question clearly and concisely.

    Notes:
    {context}

    Question:
    {user_question}
    """

    # Query the LLM
    response = llm.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)

    st.session_state.chat_history.append(("assistant", response_text))

# Display chat history in bubbles
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)
