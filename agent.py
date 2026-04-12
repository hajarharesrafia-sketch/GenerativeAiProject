"""
agent.py — Agent LangChain avec Groq + outils juridiques (Partie 2)
Style inspiré du notebook du professeur
Utilise initialize_agent + Groq (Llama3) au lieu de GPT-2
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_classic.agents import initialize_agent, AgentType

from tools import ALL_TOOLS

# Charger les variables d'environnement
load_dotenv()

# ===========================================================
# Configuration
# ===========================================================
FAISS_DIR = Path("faiss_index")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ===========================================================
# LLM — Groq (Llama3)
# ===========================================================

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    groq_api_key=GROQ_API_KEY,
)


# ===========================================================
# Outil RAG — documents internes (Partie 1)
# ===========================================================

def load_rag_tool() -> Tool:
    """Charge l'index FAISS et expose le RAG comme outil."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = db.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template(
        "Réponds à la question en français en te basant sur le contexte.\n"
        "Contexte: {context}\nQuestion: {question}\nRéponse:"
    )

    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return Tool(
        name="recherche_documents_internes",
        func=lambda q: qa_chain.invoke(q),
        description=(
            "Utilise cet outil pour répondre aux questions basées sur les documents "
            "juridiques internes : droits des salariés, convention collective, "
            "obligations sociales du spectacle vivant et enregistré. "
            "À utiliser en priorité pour toute question sur ces documents spécifiques."
        ),
    )


# ===========================================================
# Construction de l'agent
# ===========================================================

def build_agent():
    """Construit et retourne l'agent."""
    print("[INFO] Chargement de l'index FAISS (RAG)...")
    rag_tool = load_rag_tool()

    tools = [rag_tool] + ALL_TOOLS

    print("[INFO] Création de l'agent...")
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    print("[SUCCESS] Agent juridique prêt.")
    return agent


# ===========================================================
# Test CLI
# ===========================================================

if __name__ == "__main__":
    agent = build_agent()

    print("\n" + "=" * 60)
    print("Assistant Juridique Intelligent — Mode Terminal")
    print("Tapez 'quitter' pour arrêter.")
    print("=" * 60 + "\n")

    exemples = [
        "Quels sont mes droits en cas de licenciement ?",
        "Mon préavis commence le 15/04/2025, quelle est la date de fin pour 2 mois ?",
        "Quelles sont les règles sur les congés payés ?",
        "Quelles sont les obligations d'un entrepreneur de spectacles vivants ?",
    ]
    print("Exemples de questions :")
    for ex in exemples:
        print(f"  → {ex}")
    print()

    while True:
        user_input = input("Vous : ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quitter", "exit", "quit"}:
            print("Au revoir !")
            break

        try:
            response = agent.invoke({"input": user_input})
            print(f"\nAssistant : {response['output']}\n")
        except Exception as e:
            print(f"[ERREUR] {e}\n")
