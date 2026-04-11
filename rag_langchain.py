from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =========================
# Configuration
# =========================
DATA_DIR = Path("Data")
FAISS_DIR = Path("faiss_index")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SUPPORTED_EXTENSIONS = {".pdf"}


def find_documents(data_dir: Path) -> List[Path]:
    """Return all supported documents found in the data directory."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Le dossier '{data_dir}' est introuvable. Crée-le et ajoute tes PDF dedans."
        )

    files = [
        path for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise FileNotFoundError(
            f"Aucun fichier PDF trouvé dans '{data_dir}'. Ajoute au moins un document."
        )

    return sorted(files)


def load_pdf_documents(pdf_paths: List[Path]) -> List[Document]:
    """Load all PDF pages as LangChain documents."""
    documents: List[Document] = []

    for pdf_path in pdf_paths:
        print(f"[INFO] Chargement du PDF : {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        pdf_docs = loader.load()

        # enrich metadata with source filename
        for doc in pdf_docs:
            doc.metadata["source"] = pdf_path.name

        documents.extend(pdf_docs)

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(f"[INFO] Nombre total de chunks générés : {len(chunks)}")
    return chunks


def build_vector_store(chunks: List[Document], save_dir: Path) -> None:
    """Create and save a FAISS vector store locally."""
    print(f"[INFO] Chargement du modèle d'embeddings : {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("[INFO] Création de l'index FAISS...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    save_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(save_dir))
    print(f"[SUCCESS] Index FAISS sauvegardé dans : {save_dir.resolve()}")


def main() -> None:
    print("=" * 60)
    print("Construction de l'index RAG avec LangChain + FAISS")
    print("=" * 60)

    pdf_paths = find_documents(DATA_DIR)
    print(f"[INFO] {len(pdf_paths)} PDF(s) détecté(s).")

    documents = load_pdf_documents(pdf_paths)
    print(f"[INFO] {len(documents)} pages chargées.")

    chunks = split_documents(documents)
    build_vector_store(chunks, FAISS_DIR)

    print("\n[TERMINE] Ton index RAG est prêt.")
    print("Tu peux maintenant lancer l'application Chainlit pour interroger les documents.")


if __name__ == "__main__":
    main()

from langchain_classic.chains import RetrievalQA
from transformers import pipeline as hf_pipeline
from langchain_community.llms import HuggingFacePipeline


def ask_question(query: str) -> str:
    """Pose une question au système RAG."""

    # Charger embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Charger index FAISS
    db = FAISS.load_local(
    str(FAISS_DIR),
    embeddings,
    allow_dangerous_deserialization=True
)

    retriever = db.as_retriever()

    # Modèle LLM
    pipe = hf_pipeline("text-generation", model="gpt2")

    llm = HuggingFacePipeline(pipeline=pipe)

    # Chaîne RAG
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    result = qa.run(query)

    return result