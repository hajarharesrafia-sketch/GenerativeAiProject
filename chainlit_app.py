"""
chainlit_app -version finale — Interface Chainlit (Partie 3 + 4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Une mémoire distincte est créée par session utilisateur
grâce à cl.user_session — chaque utilisateur a son propre
historique de conversation.
"""

import chainlit as cl
from integration_memo import IntegratedAssistant

# ── Initialisation unique au démarrage du serveur ────────────
assistant = IntegratedAssistant()

ROUTE_LABELS = {
    "rag":          "📄 *Réponse basée sur les documents internes (RAG)*",
    "agent":        "🔧 *Réponse via un outil externe (Agent)*",
    "conversation": "💬 *Conversation générale*",
}


@cl.on_chat_start
async def on_chat_start():
    # ← NOUVEAU : une mémoire indépendante par session utilisateur
    from memory import ConversationMemory
    cl.user_session.set("memory", ConversationMemory(max_turns=10))

    await cl.Message(
        content=(
            "👋 **Bonjour ! Je suis votre assistant juridique intelligent.**\n\n"
            "Je peux :\n"
            "- 📄 Répondre à des questions sur vos **documents internes** "
            "(droits des salariés, convention collective, spectacle vivant…)\n"
            "- 🔧 Utiliser des **outils** : calcul de délais, Légifrance, recherche web\n"
            "- 💬 Discuter normalement\n\n"
            "Je me souviens du contexte de notre conversation. "
            "Tapez **reset** pour effacer la mémoire.\n\n"
            "Posez-moi votre question !"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    question = message.content.strip()

    # Commande reset
    if question.lower() == "reset":
        memory = cl.user_session.get("memory")
        memory.clear()
        await cl.Message(content="🗑️ Mémoire effacée. Nouvelle conversation !").send()
        return

    # Récupérer la mémoire de la session
    memory = cl.user_session.get("memory")

    # Injecter la mémoire de session dans l'assistant pour ce tour
    async with cl.Step(name="Analyse de la question...") as step:
        # On utilise directement les fonctions du module en passant la mémoire de session
        from integration_memo import (
            route_question, RouteType,
            answer_rag, answer_agent, answer_conversation,
        )

        route = route_question(question)
        step.output = f"Route : **{route.value.upper()}**"

        if route == RouteType.RAG:
            response = answer_rag(question, assistant.retriever, memory)
        elif route == RouteType.AGENT:
            response = answer_agent(question, assistant.agent, memory)
        else:
            response = answer_conversation(question, memory)

        # Sauvegarder dans la mémoire de session
        memory.add(question, response)

    route_label   = ROUTE_LABELS.get(route.value, f"*Route : {route.value}*")
    full_response = f"{route_label}\n\n{response}"

    await cl.Message(content=full_response).send()
