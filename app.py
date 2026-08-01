import os
import pickle
import faiss
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================
# API KEY
# ==========================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY belum diatur pada Streamlit Secrets.")
    st.stop()

# ==========================
# LOAD MODEL
# ==========================

@st.cache_resource
def load_model():
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
    index = faiss.read_index("index.faiss")

    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )

    return embedding_model, index, chunks, llm


embedding_model, index, chunks, llm = load_model()


# ==========================
# RETRIEVE
# ==========================

def retrieve(question, top_k=3):

    query_embedding = embedding_model.encode(
        "query: " + question,
        normalize_embeddings=True,
        convert_to_tensor=False
    ).astype("float32")

    scores, indices = index.search(
        np.array([query_embedding]),
        top_k
    )

    docs = []

    for score, idx in zip(scores[0], indices[0]):
        docs.append({
            "page": chunks[idx]["page"],
            "score": float(score),
            "text": chunks[idx]["text"]
        })

    return docs


# ==========================
# CHATBOT
# ==========================

def chatbot(question):

    docs = retrieve(question)

    context = "\n\n".join(
        doc["text"] for doc in docs
    )

    prompt = f"""
Kamu adalah chatbot akademik Program Studi Statistika Universitas Islam Indonesia.

Jawablah pertanyaan HANYA berdasarkan konteks berikut.

Jika jawaban tidak ditemukan pada konteks, katakan:

"Informasi tidak ditemukan pada dokumen."

====================
KONTEKS
====================

{context}

====================
PERTANYAAN
====================

{question}

====================
JAWABAN
====================
"""

    try:

        response = llm.invoke(prompt)

        if isinstance(response.content, list):

            answer = ""

            for item in response.content:
                if isinstance(item, dict):
                    answer += item.get("text", "")

        else:
            answer = str(response.content)

    except Exception as e:

        answer = f"Terjadi kesalahan:\n\n{e}"

    references = "\n".join(
        f"• Halaman {doc['page']} | Similarity : {doc['score']:.4f}"
        for doc in docs
    )

    return answer, references


# ==========================
# STREAMLIT UI
# ==========================

st.set_page_config(
    page_title="Chatbot Buku Panduan Akademik",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Chatbot Buku Panduan Akademik")

st.markdown("""
**Retrieval-Augmented Generation (RAG)**

📄 Dokumen : Buku Panduan Akademik Program Studi Statistika UII

🧠 Embedding : intfloat/multilingual-e5-base

🗂️ Vector Database : FAISS

🤖 LLM : Gemini 3.1 Flash Lite
""")

question = st.text_input(
    "Masukkan Pertanyaan",
    placeholder="Contoh: Apa visi Program Studi Statistika?"
)

if st.button("Kirim"):

    if question.strip() == "":
        st.warning("Masukkan pertanyaan terlebih dahulu.")
    else:

        with st.spinner("Sedang mencari jawaban..."):

            answer, references = chatbot(question)

        st.subheader("📌 Jawaban")
        st.write(answer)

        st.divider()

        st.subheader("📖 Referensi")
        st.write(references)
