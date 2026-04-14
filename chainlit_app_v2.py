"""
chainlit_app_v2.py — Interface Chainlit avec Agent juridique (Partie 4)
Mémoire conversationnelle par session utilisateur
"""

from __future__ import annotations

import chainlit as cl
from agent import build_agent



@cl.on_chat_start
async def on_chat_start():
    agent = build_agent()
    cl.user_session.set("agent", agent)

    await cl.Message(
        content=(
            "⚖️ **Assistant Juridique Intelligent**\n\n"
            "Je suis spécialisé en **droit du travail français**. Je peux vous aider sur :\n\n"
            "📄 Questions sur vos **documents internes** (droits du salarié, "
            "conventions collectives, obligations sociales du spectacle)\n"
            "📅 **Calcul de délais juridiques** (préavis, période d'essai, "
            "délais de recours...)\n"
            "📚 **Recherche d'articles de loi** dans le Code du travail (Légifrance)\n"
            "🔍 **Actualités juridiques** et jurisprudences récentes\n\n"
            "Comment puis-je vous aider ?"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    user_input = message.content.strip()

    async with cl.Step(name="Analyse en cours...") as step:
        try:
            response = await agent.ainvoke({"input": user_input})  # ✅ async
            answer = response.get("output", "Je n'ai pas pu générer une réponse.")
        except Exception as e:
            answer = f"Une erreur s'est produite : {e}"

        step.output = answer

    await cl.Message(content=answer).send()
