## 📚 Lecture Notes Assistant

A Streamlit-powered RAG app that lets you **chat with your lecture notes**.  
Upload up to **3 PDFs per session**, have them embedded into **Pinecone**, and ask questions powered by **OpenAI’s GPT-4o-mini**.

---

## 🛠️ Tech Stack
- [Streamlit] – frontend UI
- [LangChain] – document loading, chunking, and retrieval
- [OpenAI] – embeddings + GPT model
- [Pinecone] – vector database

---

## 🧪 Usage
- Upload 1–3 PDF lecture notes.
- Ask a question in the chat box.
- The assistant retrieves the most relevant chunks and answers using GPT-4o-mini.
- Notes are private to your session and not visible to others.

---

## 📌 Persistence Notes
- Uploaded PDFs are stored permanently in Pinecone.
- However, each session gets a random session ID. If you close the app and reopen, you’ll get a new session and won’t see your old notes.
- Your data still exists in Pinecone, but it’s orphaned (not connected to your new session).
- Future improvement: add login or session-resume to reuse old notes.

---
