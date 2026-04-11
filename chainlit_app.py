import chainlit as cl
import re
import torch

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# ============================
# CONFIGURATION
# ============================

FAISS_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
QA_MODEL = "etalab-ia/camembert-base-squadFR-fquad-piaf"

# ============================
# LOAD MODELS
# ============================

print("[INFO] Chargement des embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print("[INFO] Chargement de l'index FAISS...")
db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 30})

print(f"[INFO] Chargement du modele QA : {QA_MODEL}")
qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
print("[INFO] Modeles charges avec succes!")


# Words to ignore when extracting topic keywords
STOP_WORDS = {
    "les", "des", "du", "de", "la", "le", "un", "une", "en", "cas",
    "est", "sont", "quels", "quel", "quelle", "quelles", "que", "qui",
    "comment", "pourquoi", "dans", "par", "pour", "sur", "avec", "aux",
    "son", "ses", "leur", "ce", "cette", "ces", "et", "ou", "mais",
    "donc", "car", "ni", "ne", "pas", "plus", "droits", "droit",
    "salarié", "salarie", "salaries", "salariés", "employeur",
    "travail", "contrat", "article", "convention", "collective",
}


def get_topic_words(question: str) -> list:
    """Extract the main topic words from a question (stems for matching)."""
    words = re.findall(r"[a-zàâäéèêëïîôùûüÿç]+", question.lower())
    topics = [w for w in words if w not in STOP_WORDS and len(w) > 3]
    # Use word stems (first 6+ chars) to match variants
    # e.g. "licenciement" -> "licenci" matches licencié, licencier, etc.
    stems = []
    for w in topics:
        stem = w[:min(len(w), max(6, len(w) - 3))]
        if stem not in stems:
            stems.append(stem)
    return stems


def extract_answer(question: str, context: str) -> dict:
    """Extract answer span + surrounding context from a chunk."""
    inputs = qa_tokenizer(
        question, context,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    with torch.no_grad():
        outputs = qa_model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1

    start_score = outputs.start_logits[0][start_idx].item()
    end_score = outputs.end_logits[0][end_idx - 1].item()
    confidence = (start_score + end_score) / 2

    answer = qa_tokenizer.decode(
        inputs["input_ids"][0][start_idx:end_idx],
        skip_special_tokens=True
    )

    # Show the full chunk content — no truncation
    answer_clean = answer.strip()
    extended = context.strip()

    return {
        "answer": answer_clean,
        "extended": extended,
        "confidence": confidence,
    }


# ============================
# CHAINLIT EVENTS
# ============================

@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 Bonjour ! Je suis ton assistant juridique.\n\n"
                "Pose-moi une question sur :\n"
                "- Le droit du travail\n"
                "- Les conventions collectives\n"
                "- Les droits et obligations des salaries\n\n"
                "Je cherche dans les documents et je te donne les passages pertinents."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    question = message.content

    try:
        docs = await cl.make_async(retriever.invoke)(question)

        if not docs:
            await cl.Message(content="Aucun passage pertinent trouve.").send()
            return

        # Get topic words to filter irrelevant chunks
        topics = get_topic_words(question)

        # Extract answers, filtering by topic relevance
        results = []
        seen_answers = set()
        for doc in docs:
            # Skip chunks that don't mention any topic word
            chunk_lower = doc.page_content.lower()
            if topics and not any(t in chunk_lower for t in topics):
                continue

            r = extract_answer(question, doc.page_content)
            if (r["confidence"] < -3
                    or len(r["answer"]) < 3
                    or r["answer"].lower() in seen_answers):
                continue
            seen_answers.add(r["answer"].lower())
            results.append({
                **r,
                "source": doc.metadata.get("source", "inconnu"),
                "page": doc.metadata.get("page", "?"),
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)

        if not results:
            await cl.Message(
                content="Je n'ai pas trouve de reponse precise. "
                        "Essaie de reformuler ta question."
            ).send()
            return

        # Build response
        response = f"## Resultats pour : *{question}*\n\n"

        for i, r in enumerate(results[:4], 1):
            response += f"### 📌 Extrait {i}"
            response += f" — 📄 {r['source']} (p.{r['page']})\n\n"
            response += f"> {r['extended']}\n\n"

    except Exception as e:
        response = f"Erreur : {str(e)}"

    print("QUESTION:", question)
    await cl.Message(content=response).send()