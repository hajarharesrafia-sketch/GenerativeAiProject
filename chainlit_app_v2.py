"""
chainlit_app_v2.py — Interface Chainlit avec Agent juridique (Partie 2)
"""

from __future__ import annotations

import chainlit as cl
from agent import build_agent

agent_executor = build_agent()


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="⚖️ Je suis spécialisé en **droit du travail français**, posez-moi une question !"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content.strip()

    async with cl.Step(name="Analyse en cours...") as step:
        try:
            response = agent_executor.invoke({"input": user_input})
            answer = response.get("output", "Je n'ai pas pu générer une réponse.")
        except Exception as e:
            answer = f"Une erreur s'est produite : {e}"

        step.output = answer

    await cl.Message(content=answer).send()
