"""
chainlit_app_part3.py — Interface Chainlit pour la Partie 3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Interface conversationnelle qui affiche la route utilisée
(RAG / Agent / Conversation) pour chaque réponse.
"""

import chainlit as cl
from integration import IntegratedAssistant

# ─── Initialisation unique au démarrage du serveur ───────────────
assistant = IntegratedAssistant()

# Labels affichés à l'utilisateur selon la route choisie
ROUTE_LABELS = {
    "rag":          "📄 *Réponse basée sur les documents internes (RAG)*",
    "agent":        "🔧 *Réponse via un outil externe (Agent)*",
    "conversation": "💬 *Conversation générale*",
}


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "👋 **Bonjour ! Je suis votre assistant juridique intelligent.**\n\n"
            "Je peux :\n"
            "- 📄 Répondre à des questions sur vos **documents internes** "
            "(droits des salariés, convention collective, spectacle vivant…)\n"
            "- 🔧 Utiliser des **outils** : calcul de délais, Légifrance, recherche web\n"
            "- 💬 Discuter normalement\n\n"
            "Posez-moi votre question !"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    question = message.content.strip()

    # Indicateur de chargement
    async with cl.Step(name="Analyse de la question...") as step:
        result   = assistant.answer(question)
        route    = result["route"]
        response = result["response"]
        step.output = f"Route sélectionnée : **{route.upper()}**"

    # Label de route + réponse
    route_label = ROUTE_LABELS.get(route, f"*Route : {route}*")
    full_response = f"{route_label}\n\n{response}"

    await cl.Message(content=full_response).send()