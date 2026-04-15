"""
part3_integration.py — Partie 3 + Partie 4 (mémoire conversationnelle)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Routeur intelligent qui combine :
  • RAG     → si la question concerne les documents internes
  • Agent   → si la question peut être répondue par un outil
  • LLM     → sinon (conversation normale)

+ Mémoire conversationnelle (Partie 4) :
  L'historique des échanges est injecté dans chaque appel LLM
  pour que l'assistant se souvienne du contexte.
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
from langchain_classic.agents import initialize_agent, AgentType

from tools import ALL_TOOLS
from memory import ConversationMemory          # ← NOUVEAU (Partie 4)

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

    if "rag" in raw:
        return RouteType.RAG
    if "agent" in raw:
        return RouteType.AGENT
    return RouteType.CONVERSATION


# ═══════════════════════════════════════════════════════════════
# Voie 1 — RAG avec citations + historique (Partie 4)
# ═══════════════════════════════════════════════════════════════

RAG_PROMPT_TEMPLATE = """Tu es un assistant juridique expert.
Réponds à la question en français en te basant UNIQUEMENT sur le contexte fourni.
Tu dois OBLIGATOIREMENT inclure des citations en mentionnant la source (nom du fichier et page).

Format des citations : [Source : NomDuFichier, page X]

Si la réponse ne se trouve pas dans le contexte, dis-le clairement.

{history_section}
Contexte :
{context}

Question : {question}

Réponse (avec citations) :"""


def format_docs_with_citations(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "inconnu")
        page   = doc.metadata.get("page", "?")
        parts.append(f"[Document {i} — Source : {source}, page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_rag_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 4})


def answer_rag(question: str, retriever, memory: ConversationMemory) -> str:
    """Répond via RAG en injectant l'historique dans le prompt."""
    docs     = retriever.invoke(question)
    context  = format_docs_with_citations(docs)

    history_text = memory.get_history_as_text()
    history_section = (
        f"Historique de la conversation :\n{history_text}\n\n"
        if history_text else ""
    )

    prompt_text = RAG_PROMPT_TEMPLATE.format(
        history_section=history_section,
        context=context,
        question=question,
    )

    return llm.invoke([HumanMessage(content=prompt_text)]).content


# ═══════════════════════════════════════════════════════════════
# Voie 2 — Agent avec outils + historique (Partie 4)
# ═══════════════════════════════════════════════════════════════

def build_agent():
    return initialize_agent(
        tools=ALL_TOOLS,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )


def answer_agent(question: str, agent, memory: ConversationMemory) -> str:
    """Appelle l'agent en enrichissant la question avec l'historique."""
    history_text = memory.get_history_as_text()

    enriched_input = (
        f"Contexte de la conversation précédente :\n{history_text}\n\nNouvelle question : {question}"
        if history_text else question
    )

    result = agent.invoke({"input": enriched_input})
    return result.get("output", str(result))


# ═══════════════════════════════════════════════════════════════
# Voie 3 — Conversation normale + historique (Partie 4)
# ═══════════════════════════════════════════════════════════════

CONVERSATION_SYSTEM_PROMPT = """Tu es un assistant juridique intelligent et sympathique.
Pour les questions simples ou les salutations, réponds de façon naturelle et chaleureuse en français.
Tu peux rappeler à l'utilisateur que tu peux répondre à des questions sur le droit du travail
ou effectuer des calculs de délais juridiques.
Tiens compte de l'historique de la conversation pour répondre de façon cohérente."""


def answer_conversation(question: str, memory: ConversationMemory) -> str:
    """Répond directement avec le LLM en injectant l'historique."""
    messages = [SystemMessage(content=CONVERSATION_SYSTEM_PROMPT)]
    messages += memory.get_history_as_messages()   # ← historique LangChain
    messages.append(HumanMessage(content=question))
    return llm.invoke(messages).content


# ═══════════════════════════════════════════════════════════════
# Classe principale — Assistant Intégré (Partie 3 + 4)
# ═══════════════════════════════════════════════════════════════

class IntegratedAssistant:
    """
    Assistant intelligent avec mémoire conversationnelle (Partie 4).
    Route chaque question et conserve l'historique des échanges.
    """

    def __init__(self, max_turns: int = 10):
        print("[INFO] Initialisation de l'assistant intégré...")

        print("[INFO] Chargement du pipeline RAG...")
        self.retriever = build_rag_retriever()

        print("[INFO] Initialisation de l'agent...")
        self.agent = build_agent()

        self.memory = ConversationMemory(max_turns=max_turns)   # ← NOUVEAU
        print("[SUCCESS] Assistant prêt !\n")

    def answer(self, question: str) -> dict:
        """
        Traite une question, utilise la mémoire pour le contexte,
        puis sauvegarde le nouvel échange.
        """
        print(f"\n[ROUTEUR] Analyse : '{question}'")
        print(f"[MÉMOIRE] {self.memory}")

        route = route_question(question)
        print(f"[ROUTEUR] → Route : {route.value.upper()}")

        if route == RouteType.RAG:
            response = answer_rag(question, self.retriever, self.memory)

        elif route == RouteType.AGENT:
            response = answer_agent(question, self.agent, self.memory)

        else:
            response = answer_conversation(question, self.memory)

        self.memory.add(question, response)   # ← NOUVEAU : sauvegarde

        return {"route": route.value, "response": response}

    def reset_memory(self) -> None:
        self.memory.clear()
        print("[MÉMOIRE] Historique effacé.")


# ═══════════════════════════════════════════════════════════════
# Mode terminal — test CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    assistant = IntegratedAssistant()

    print("=" * 65)
    print("  Assistant Juridique — Partie 3 + 4 (avec mémoire)")
    print("  Tapez 'quitter' pour arrêter, 'reset' pour effacer la mémoire.")
    print("=" * 65 + "\n")

    while True:
        user_input = input("Vous : ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quitter", "exit", "quit"}:
            print("Au revoir !")
            break
        if user_input.lower() == "reset":
            assistant.reset_memory()
            print("Mémoire effacée.\n")
            continue

        result = assistant.answer(user_input)
        route_label = {
            "rag":          "📄 RAG (documents internes)",
            "agent":        "🔧 Agent (outils)",
            "conversation": "💬 Conversation normale",
        }.get(result["route"], result["route"])

        print(f"\n[{route_label}]")
        print(f"Assistant : {result['response']}\n")
