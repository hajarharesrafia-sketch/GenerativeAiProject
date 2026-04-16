import chainlit as cl
from langchain_classic.memory import ConversationBufferWindowMemory
from integration_memo import IntegratedAssistant

assistant = IntegratedAssistant()

ROUTE_LABELS = {
    "rag": "📄 *Réponse basée sur les documents internes (RAG)*",
    "agent": "🔧 *Réponse via un outil externe (Agent)*",
    "conversation": "💬 *Conversation générale*",
}

@cl.on_chat_start
async def on_chat_start():
    memory = ConversationBufferWindowMemory(
        k=3,
        memory_key="history",
        input_key="input",
        output_key="output",
        return_messages=False,
    )
    cl.user_session.set("memory", memory)

    await cl.Message(
        content=(
            "👋 **Bonjour ! Je suis votre assistant juridique intelligent.**\n\n"
            "Je peux :\n"
            "- 📄 Répondre à des questions sur vos **documents internes** "
            "(droits des salariés, convention collective, spectacle vivant…)\n"
            "- 🔧 Utiliser des **outils** : calcul de délais, Légifrance, recherche web\n"
            "- 💬 Discuter normalement\n\n"
            "Tapez **reset** pour effacer la mémoire.\n\n"
            "Posez-moi votre question !"
        )
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    question = message.content.strip()
    memory = cl.user_session.get("memory")

    if question.lower() == "reset":
        memory.clear()
        await cl.Message(content="🗑️ Mémoire effacée.").send()
        return

    assistant.set_memory(memory)
    result = assistant.answer(question)

    route_label = ROUTE_LABELS.get(result["route"], result["route"])
    await cl.Message(content=f"{route_label}\n\n{result['response']}").send()
