"""
part3_integration.py — Partie 3 : Intégration finale
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Routeur intelligent qui combine :
  • RAG  → si la question concerne les documents internes
  • Agent (outils) → si la question peut être répondue par un outil
  • LLM direct → sinon (conversation normale)

Utilise Groq (Llama3) + LangChain, style du projet groupe.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import Tool
from langchain_classic.agents import initialize_agent, AgentType

from tools import ALL_TOOLS

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

FAISS_DIR       = Path("faiss_index")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")

# ═══════════════════════════════════════════════════════════════
# LLM — Groq (Llama3)
# ═══════════════════════════════════════════════════════════════

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=GROQ_API_KEY,
)


# ═══════════════════════════════════════════════════════════════
# Enum : types de routes possibles
# ═══════════════════════════════════════════════════════════════

class RouteType(str, Enum):
    RAG          = "rag"
    AGENT        = "agent"
    CONVERSATION = "conversation"


# ═══════════════════════════════════════════════════════════════
# Routeur — décide quelle voie emprunter
# ═══════════════════════════════════════════════════════════════

ROUTER_SYSTEM_PROMPT = """Tu es un routeur pour un assistant juridique intelligent.
Ta tâche est de classer la question de l'utilisateur dans l'UNE de ces trois catégories :

1. "rag"          → La question porte sur des documents internes : droits des salariés,
                     convention collective, obligations sociales, spectacle vivant,
                     congés payés, licenciement, préavis, période d'essai, CDDU, etc.

2. "agent"        → La question nécessite un outil externe :
                     calcul de délai/date, recherche d'un article de loi sur Légifrance,
                     recherche web d'actualité juridique, météo, calcul, etc.

3. "conversation" → Salutation, question générale sans lien avec les documents ou les outils,
                     bavardage (ex. : "Bonjour", "Merci", "Comment vas-tu ?").

Réponds UNIQUEMENT avec l'un de ces trois mots exacts : rag, agent, conversation.
Ne fournis aucune explication.
"""


def route_question(question: str) -> RouteType:
    """Détermine quelle voie utiliser pour répondre à la question."""
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    raw = llm.invoke(messages).content.strip().lower()

    # Sécurité : si la réponse ne correspond pas exactement, on déduit
    if "rag" in raw:
        return RouteType.RAG
    if "agent" in raw:
        return RouteType.AGENT
    return RouteType.CONVERSATION


# ═══════════════════════════════════════════════════════════════
# Voie 1 — RAG avec citations
# ═══════════════════════════════════════════════════════════════

RAG_PROMPT_TEMPLATE = """Tu es un assistant juridique expert.
Réponds à la question en français en te basant UNIQUEMENT sur le contexte fourni.
Tu dois OBLIGATOIREMENT inclure des citations en mentionnant la source (nom du fichier et page).

Format des citations : [Source : NomDuFichier, page X]

Si la réponse ne se trouve pas dans le contexte, dis-le clairement.

Contexte :
{context}

Question : {question}

Réponse (avec citations) :"""


def format_docs_with_citations(docs) -> str:
    """Formate les documents récupérés en ajoutant les métadonnées de source."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "inconnu")
        page   = doc.metadata.get("page", "?")
        parts.append(
            f"[Document {i} — Source : {source}, page {page}]\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def build_rag_chain():
    """Construit la chaîne RAG avec support des citations."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = db.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    rag_chain = (
        {
            "context":  retriever | format_docs_with_citations,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# ═══════════════════════════════════════════════════════════════
# Voie 2 — Agent avec outils
# ═══════════════════════════════════════════════════════════════

def build_agent():
    """Construit l'agent LangChain avec tous les outils (Partie 2)."""
    agent = initialize_agent(
        tools=ALL_TOOLS,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent


# ═══════════════════════════════════════════════════════════════
# Voie 3 — Conversation normale (LLM direct)
# ═══════════════════════════════════════════════════════════════

CONVERSATION_SYSTEM_PROMPT = """Tu es un assistant juridique intelligent et sympathique.
Pour les questions simples ou les salutations, réponds de façon naturelle et chaleureuse en français.
Tu peux rappeler à l'utilisateur que tu peux répondre à des questions sur le droit du travail
ou effectuer des calculs de délais juridiques."""


def answer_conversation(question: str) -> str:
    """Répond directement avec le LLM, sans RAG ni outils."""
    messages = [
        SystemMessage(content=CONVERSATION_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    return llm.invoke(messages).content


# ═══════════════════════════════════════════════════════════════
# Classe principale — Assistant Intégré (Partie 3)
# ═══════════════════════════════════════════════════════════════

class IntegratedAssistant:
    """
    Assistant intelligent qui route automatiquement chaque question
    vers le bon système : RAG, Agent ou conversation normale.
    """

    def __init__(self):
        print("[INFO] Initialisation de l'assistant intégré (Partie 3)...")

        print("[INFO] Chargement du pipeline RAG...")
        self.rag_chain = build_rag_chain()

        print("[INFO] Initialisation de l'agent...")
        self.agent = build_agent()

        print("[SUCCESS] Assistant prêt !\n")

    def answer(self, question: str) -> dict:
        """
        Traite une question et retourne la réponse avec le type de route utilisée.

        Retourne :
            {
                "route":    "rag" | "agent" | "conversation",
                "response": str,
            }
        """
        print(f"\n[ROUTEUR] Analyse de la question : '{question}'")

        route = route_question(question)
        print(f"[ROUTEUR] → Route sélectionnée : {route.value.upper()}")

        if route == RouteType.RAG:
            response = self.rag_chain.invoke(question)

        elif route == RouteType.AGENT:
            result   = self.agent.invoke({"input": question})
            response = result.get("output", str(result))

        else:  # CONVERSATION
            response = answer_conversation(question)

        return {"route": route.value, "response": response}


# ═══════════════════════════════════════════════════════════════
# Mode terminal — test CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    assistant = IntegratedAssistant()

    print("=" * 65)
    print("  Assistant Juridique Intelligent — Partie 3 (Intégration)")
    print("  Tapez 'quitter' pour arrêter.")
    print("=" * 65)

    exemples = [
        "Bonjour !",
        "Quelle est la politique de congés dans les documents ?",
        "Mon préavis commence le 15/04/2025, quelle est la date de fin pour 2 mois ?",
        "Quels sont les droits en cas de licenciement ?",
        "Recherche l'article sur les congés payés sur Légifrance.",
    ]
    print("\nExemples de questions pour tester les 3 routes :")
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

        result = assistant.answer(user_input)
        route_label = {
            "rag":          "📄 RAG (documents internes)",
            "agent":        "🔧 Agent (outils)",
            "conversation": "💬 Conversation normale",
        }.get(result["route"], result["route"])

        print(f"\n[{route_label}]")
        print(f"Assistant : {result['response']}\n")