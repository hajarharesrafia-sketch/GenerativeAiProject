"""
chainlit_app_v2.py — Interface Chainlit avec Agent juridique (Partie 4)
S'appuie sur IntegratedAssistant (Partie 3) + mémoire conversationnelle par session
"""
from __future__ import annotations

import chainlit as cl
from integration import IntegratedAssistant

# Labels affichés à l'utilisateur selon la route choisie
ROUTE_LABELS = {
    "rag":          "📄 *Réponse basée sur les documents internes (RAG)*",
    "agent":        "🔧 *Réponse via un outil externe (Agent)*",
    "conversation": "💬 *Conversation générale*",
}

@cl.on_chat_start
async def on_chat_start():
    assistant = IntegratedAssistant()
    cl.user_session.set("assistant", assistant)

    await cl.Message(
        content="⚖️ Je suis spécialisé en **droit du travail français**, posez-moi une question !"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    assistant = cl.user_session.get("assistant")
    question = message.content.strip()

    async with cl.Step(name="Analyse de la question...") as step:
        result   = assistant.answer(question)
        route    = result["route"]
        response = result["response"]
        step.output = f"Route sélectionnée : **{route.upper()}**"

    route_label = ROUTE_LABELS.get(route, f"*Route : {route}*")
    full_response = f"{route_label}\n\n{response}"

    await cl.Message(
    content="**Assistant Juridique** — Posez-moi votre question sur le droit du travail français."
).send()
