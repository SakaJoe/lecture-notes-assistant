import tempfile
import streamlit as st
import uuid
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# Assign a unique session_id for each user (persists during their session)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]  
INDEX_NAME = "rag-index"


# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)


if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,
)

vector_store = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)


# Streamlit UI
st.set_page_config(page_title="Lecture Notes Assistant", layout="wide")
st.title("Lecture Notes Q&A Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------
# Upload PDF Section
# -------------------
uploaded_file = st.file_uploader("Upload lecture notes (PDF)", type=["pdf"])

if uploaded_file is not None:
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

    # Add documents to Pinecone
    vector_store.add_documents(documents)
    # Add metadata: source file + user session
    for doc in documents:
        doc.metadata["source"] = uploaded_file.name
        doc.metadata["user"] = st.session_state.session_id

    vector_store.add_documents(documents)
    st.success(f"Added {len(documents)} chunks from {uploaded_file.name} to Pinecone DB (private to your session)")

# -------------------
# Chat Interface
# -------------------
st.subheader("Chat with your notes")

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
