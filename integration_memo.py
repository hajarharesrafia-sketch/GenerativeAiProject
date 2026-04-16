"""
integration_memo.py — Partie 3 + Partie 4
Routage intelligent + mémoire conversationnelle LangChain
==========================================================

Routes :
- RAG          : questions sur le CONTENU des documents internes
- AGENT        : questions nécessitant un outil externe (calcul, Légifrance, web)
- CONVERSATION : échange général, salutations

"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.agents import initialize_agent, AgentType

from tools import ALL_TOOLS

# =============================================================
# Configuration
# =============================================================
load_dotenv()

FAISS_DIR       = Path("faiss_index")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL      = "llama-3.3-70b-versatile"

# =============================================================
# LLM
# =============================================================
llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    groq_api_key=GROQ_API_KEY,
)

# =============================================================
# Enum des routes
# =============================================================
class RouteType(str, Enum):
    RAG          = "rag"
    AGENT        = "agent"
    CONVERSATION = "conversation"


# =============================================================
# Routeur — VERSION CORRIGÉE
# =============================================================
# PROBLÈME D'ORIGINE : le prompt était ambigu. Les mots-clés
# comme "congés payés" ou "licenciement" apparaissaient dans
# la description RAG ET dans celle de l'agent (Légifrance),
# donc le LLM choisissait souvent RAG même pour une demande
# d'article de loi.
#
# SOLUTION : on distingue clairement l'INTENTION :
#   - RAG    = "que dit votre document sur X ?"
#   - AGENT  = "cherche/calcule/trouve l'article de loi sur X"
#   - CONVO  = salutation ou question générale
#
# On ajoute aussi des exemples concrets pour ancrer le LLM.
# =============================================================

ROUTER_SYSTEM_PROMPT = """Tu es un routeur pour un assistant juridique.
Tu dois classer la question dans UNE seule catégorie.

━━━ RÈGLES DE CLASSIFICATION ━━━

1) rag
→ L'utilisateur veut une information contenue dans les documents internes de l'entreprise.
→ La question porte sur le CONTENU d'un document déjà indexé.
Exemples :
- "Quelle est la politique de congés ?"
- "Que dit la convention collective sur le préavis ?"
- "Quelles sont les obligations du spectacle vivant selon vos documents ?"
- "Quels sont mes droits en cas de licenciement selon le document ?"

2) agent
→ L'utilisateur demande une action : calcul, recherche web, recherche d'un article officiel.
→ La réponse nécessite un OUTIL externe (Légifrance, calcul de date, recherche internet).
Exemples :
- "Donne-moi l'article de loi sur les congés payés" → Légifrance
- "Quel est le SMIC actuel ?" → recherche web
- "Mon préavis commence le 15/04/2025, quelle est la date de fin ?" → calcul de délai
- "Recherche la jurisprudence sur le licenciement abusif" → recherche web
- "Trouve l'article L1234-1 du code du travail" → Légifrance

3) conversation
→ Salutation, remerciement, question générale sans rapport avec les documents ou les outils.
Exemples :
- "Bonjour"
- "Merci"
- "Tu peux m'aider ?"
- "Comment ça marche ?"

━━━ RÈGLE PRIORITAIRE ━━━
Si la question contient : "article de loi", "article L", "code du travail",
"jurisprudence", "SMIC actuel", "recherche", "trouve", "calcule", "date de fin",
"délai de" → réponds TOUJOURS "agent".

Réponds UNIQUEMENT avec un seul mot : rag, agent, ou conversation.
"""


def route_question(question: str) -> RouteType:
    """
    Classifie la question avec une règle de sécurité supplémentaire :
    si des mots-clés d'action sont détectés, on force la route 'agent'
    sans même appeler le LLM.
    """
    q_lower = question.lower()

    # ── Règle déterministe (avant appel LLM) ──────────────────
    # Mots-clés qui signifient toujours un outil externe
    AGENT_KEYWORDS = [
        "article de loi", "article l", "code du travail",
        "légifrance", "legifrance",
        "smic actuel", "salaire minimum actuel",
        "jurisprudence",
        "calcule", "calcul", "date de fin", "date d'échéance",
        "délai de", "recherche web", "recherche sur internet",
        "météo", "meteo",
    ]
    if any(kw in q_lower for kw in AGENT_KEYWORDS):
        print(f"[ROUTEUR] Mot-clé agent détecté → AGENT (règle déterministe)")
        return RouteType.AGENT

    # ── Appel LLM pour les cas ambigus ────────────────────────
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    raw = llm.invoke(messages).content.strip().lower()

    # Tolérance : on accepte que le LLM réponde avec une phrase contenant le mot
    if "agent" in raw:
        return RouteType.AGENT
    if "rag" in raw:
        return RouteType.RAG
    return RouteType.CONVERSATION


# =============================================================
# RAG avec citations
# =============================================================
RAG_SYSTEM_PROMPT = """Tu es un assistant juridique expert.
Réponds UNIQUEMENT à partir du contexte fourni ci-dessous.

Consignes :
- Réponds en français.
- Réponse claire, utile, maximum 150 mots.
- Cite 2 à 4 points importants.
- Mentionne la source entre crochets : [Source : fichier, page X]
- Si l'information est absente du contexte, dis-le clairement.
- Ne donne JAMAIS d'information inventée.
"""


def build_rag_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(
        str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True
    )
    return db.as_retriever(search_kwargs={"k": 3})


def _safe_page(metadata: dict) -> str:
    if "page" not in metadata:
        return "?"
    try:
        return str(int(metadata["page"]) + 1)
    except Exception:
        return str(metadata["page"])


def format_docs_with_citations(docs) -> tuple[str, str]:
    context_parts = []
    source_lines  = []

    for i, doc in enumerate(docs, start=1):
        source  = doc.metadata.get("source", "inconnu")
        page    = _safe_page(doc.metadata)
        content = doc.page_content.strip()[:800]
        context_parts.append(f"[Document {i} | Source: {source} | Page: {page}]\n{content}")
        source_lines.append(f"- {source}, page {page}")

    context_text = "\n\n".join(context_parts)
    sources_text = "\n".join(dict.fromkeys(source_lines))
    return context_text, sources_text


def answer_rag(
    question: str,
    retriever,
    memory: ConversationBufferWindowMemory,
) -> str:
    docs = retriever.invoke(question)
    context_text, sources_text = format_docs_with_citations(docs)

    # Historique depuis la mémoire (texte brut)
    history = memory.load_memory_variables({}).get("history", "")

    user_prompt = (
        f"Historique récent :\n{history}\n\n"
        if history else ""
    ) + f"Contexte :\n{context_text}\n\nQuestion : {question}\n\nRéponse :"

    answer = llm.invoke([
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]).content.strip()

    if sources_text:
        answer += f"\n\nSources :\n{sources_text}"

    return answer


# =============================================================
# Agent — construit UNE SEULE fois, mémoire injectée
# =============================================================
# CORRECTION : dans l'ancienne version, l'agent était reconstruit
# à chaque appel answer_agent(), ce qui était très coûteux.
# Maintenant il est créé une seule fois dans IntegratedAssistant.__init__
# et la mémoire est mise à jour via set_memory().
#
# CORRECTION mémoire : CONVERSATIONAL_REACT_DESCRIPTION requiert
# return_messages=True. L'ancienne mémoire avait return_messages=False
# ce qui provoquait une erreur silencieuse (l'agent ignorait l'historique).
#
# CORRECTION double sauvegarde : l'agent CONVERSATIONAL_REACT sauvegarde
# lui-même dans la mémoire via memory.save_context(). Ne pas rappeler
# save_context() après dans answer(), sinon l'historique est doublé.
# =============================================================

def build_agent(memory: ConversationBufferWindowMemory):
    return initialize_agent(
        tools=ALL_TOOLS,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )


def answer_agent(question: str, agent) -> str:
    """
    Appelle l'agent. La mémoire est déjà intégrée dans l'agent
    (passée à initialize_agent), donc elle est mise à jour automatiquement.
    """
    result = agent.invoke({"input": question})
    return result.get("output", str(result)) if isinstance(result, dict) else str(result)


# =============================================================
# Conversation normale
# =============================================================
CONVERSATION_SYSTEM_PROMPT = """Tu es un assistant juridique intelligent et sympathique.
Réponds en français, de façon brève et naturelle (maximum 3 phrases).
Tu peux rappeler que tu peux répondre sur le droit du travail ou calculer des délais."""


def answer_conversation(
    question: str,
    memory: ConversationBufferWindowMemory,
) -> str:
    # Récupérer l'historique sous forme de messages LangChain
    history_messages = memory.chat_memory.messages if memory.chat_memory.messages else []
    messages = [SystemMessage(content=CONVERSATION_SYSTEM_PROMPT)]
    messages.extend(history_messages)
    messages.append(HumanMessage(content=question))
    return llm.invoke(messages).content.strip()


# =============================================================
# Mémoire partagée — factory
# =============================================================
def make_memory(k: int = 3) -> ConversationBufferWindowMemory:
    """
    Crée une mémoire LangChain compatible avec les 3 routes.

    return_messages=True  → obligatoire pour CONVERSATIONAL_REACT_DESCRIPTION
    input_key / output_key → évite l'erreur "multiple input keys"
    """
    return ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",   # clé attendue par CONVERSATIONAL_REACT
        input_key="input",
        output_key="output",
        return_messages=True,        # ← CORRECTION : False cassait l'agent
    )


# =============================================================
# Assistant intégré
# =============================================================
class IntegratedAssistant:
    """
    Assistant principal :
    - routeur intelligent (règle déterministe + LLM)
    - pipeline RAG avec citations
    - agent avec outils (construit une seule fois)
    - mémoire ConversationBufferWindowMemory partagée
    """

    def __init__(
        self,
        max_turns: int = 3,
        memory: Optional[ConversationBufferWindowMemory] = None,
    ):
        print("[INFO] Initialisation de l'assistant...")

        self.retriever = build_rag_retriever()
        self.memory    = memory or make_memory(k=max_turns)
        self.agent     = build_agent(self.memory)   # ← construit UNE SEULE fois

        print("[SUCCESS] Assistant prêt !")

    def set_memory(self, memory: ConversationBufferWindowMemory) -> None:
        """
        Injecte une nouvelle mémoire (ex. depuis Chainlit user_session).
        Reconstruit l'agent avec cette mémoire.
        """
        self.memory = memory
        self.agent  = build_agent(memory)   # ← nécessaire : l'agent garde une ref interne

    def answer(self, question: str) -> dict:
        route    = route_question(question)
        response = ""

        if route == RouteType.RAG:
            response = answer_rag(question, self.retriever, self.memory)
            # RAG n'a pas de mémoire intégrée → on sauvegarde manuellement
            self.memory.save_context(
                {"input": question},
                {"output": response},
            )

        elif route == RouteType.AGENT:
            response = answer_agent(question, self.agent)
            # L'agent CONVERSATIONAL_REACT sauvegarde lui-même → pas de save_context ici

        else:  # CONVERSATION
            response = answer_conversation(question, self.memory)
            # Conversation n'a pas de mémoire intégrée → sauvegarde manuelle
            self.memory.save_context(
                {"input": question},
                {"output": response},
            )

        return {"route": route.value, "response": response}

    def reset_memory(self) -> None:
        self.memory.clear()
        print("[MÉMOIRE] Historique effacé.")


# =============================================================
# Test terminal
# =============================================================
if __name__ == "__main__":
    assistant = IntegratedAssistant(max_turns=3)

    print("=" * 70)
    print("Assistant Juridique — RAG + Agent + Conversation + Memory Buffer")
    print("Tapez 'quitter' pour arrêter, 'reset' pour effacer la mémoire.")
    print("=" * 70)

    test_questions = [
        "Bonjour !",
        "Quelle est la politique de congés selon vos documents ?",
        "Donne-moi l'article de loi sur les congés payés",
        "Mon préavis commence le 15/04/2025, quelle est la date de fin pour 2 mois ?",
        "Quel est le SMIC actuel en France ?",
    ]
    print("\nExemples pour tester les 3 routes :")
    for q in test_questions:
        print(f"  → {q}")
    print()

    while True:
        user_input = input("\nVous : ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quitter", "quit", "exit"}:
            print("Au revoir !")
            break
        if user_input.lower() == "reset":
            assistant.reset_memory()
            print("Mémoire effacée.")
            continue

        result = assistant.answer(user_input)
        print(f"\n[ROUTE : {result['route'].upper()}]")
        print(result["response"])
