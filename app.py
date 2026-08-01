
import os
import pickle
import faiss
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY environment variable.")

embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
index = faiss.read_index("index.faiss")

with open("chunks.pkl","rb") as f:
    chunks = pickle.load(f)

embeddings = np.load("embeddings.npy")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)

def retrieve(question, top_k=3):
    query_embedding = embedding_model.encode(
        "query: " + question,
        normalize_embeddings=True,
        convert_to_tensor=False
    ).astype("float32")
    scores, indices = index.search(np.array([query_embedding]), top_k)
    docs = []
    for score, idx in zip(scores[0], indices[0]):
        docs.append({
            "page": chunks[idx]["page"],
            "score": float(score),
            "text": chunks[idx]["text"]
        })
    return docs

def chatbot(question):
    docs = retrieve(question)
    context = "\n\n".join(doc["text"] for doc in docs)
    prompt = f"""
Kamu adalah chatbot akademik Program Studi Statistika Universitas Islam Indonesia.

Jawablah pertanyaan HANYA berdasarkan konteks berikut.
Jika jawaban tidak ditemukan pada konteks, katakan:
"Informasi tidak ditemukan pada dokumen."

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN:
"""
    try:
        response = llm.invoke(prompt)
        if isinstance(response.content, list):
            answer = "".join(
                item.get("text","")
                for item in response.content
                if isinstance(item, dict)
            )
        else:
            answer = str(response.content)
    except Exception as e:
        answer = f"Terjadi kesalahan:\n\n{e}"

    refs = "\n".join(
        f"• Halaman {d['page']} | Similarity: {d['score']:.4f}"
        for d in docs
    )
    return f"""# 📌 Jawaban

{answer}

---

# 📖 Referensi

{refs}
"""

demo = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(
        lines=2,
        label="Pertanyaan",
        placeholder="Contoh: Apa visi Program Studi Statistika?"
    ),
    outputs=gr.Markdown(),
    title="📘 Chatbot Buku Panduan Akademik",
    description="""
Retrieval-Augmented Generation (RAG)

📄 Dokumen : Buku Panduan Akademik Program Studi Statistika UII
🧠 Embedding : intfloat/multilingual-e5-base
🗂️ Vector Database : FAISS
🤖 LLM : Gemini
"""
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT",7860)))
