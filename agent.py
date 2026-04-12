"""
agent.py — Agent LangChain avec outils juridiques (Partie 2)
S'intègre au pipeline RAG de rag_langchain.py (Partie 1)

L'agent décide automatiquement :
  - Question sur les documents internes → RAG
  - Délai/échéance légale → calcul_delai_juridique
  - Référence à un article de loi → recherche_legifrance
  - Actualité juridique / jurisprudence récente → web_search_juridique
  - Conversation normale → réponse directe
"""

from __future__ import annotations

from pathlib import Path

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline as hf_pipeline

from tools import ALL_TOOLS

# ===========================================================
# Configuration
# ===========================================================
FAISS_DIR = Path("faiss_index")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ===========================================================
# Chargement du LLM
# ===========================================================

def load_llm() -> HuggingFacePipeline:
    pipe = hf_pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=512,
        temperature=0.1,
    )
    return HuggingFacePipeline(pipeline=pipe)


# ===========================================================
# Outil RAG (Partie 1 de la binôme)
# ===========================================================

def load_rag_tool(llm: HuggingFacePipeline) -> Tool:
    """Charge l'index FAISS et expose le RAG comme un outil de l'agent."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = db.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )

    return Tool(
        name="recherche_documents_internes",
        func=lambda q: qa_chain.run(q),
        description=(
            "Utilise cet outil pour répondre à des questions basées sur les documents "
            "juridiques internes : droits des salariés, convention collective, "
            "obligations sociales du spectacle vivant et enregistré. "
            "À utiliser en priorité pour toute question sur ces documents spécifiques. "
            "Input : la question de l'utilisateur en français."
        ),
    )


# ===========================================================
# Prompt ReAct
# ===========================================================

REACT_PROMPT_TEMPLATE = """Tu es un assistant juridique intelligent francophone spécialisé en droit du travail français.
Tu aides les utilisateurs à comprendre leurs droits et obligations légales, que ce soit pour les salariés du privé, les conventions collectives ou les obligations du spectacle vivant et enregistré.

Tu as accès aux outils suivants :
{tools}

Règles de sélection des outils :
- Question sur les droits des salariés, conventions collectives, ou obligations sociales du spectacle → recherche_documents_internes
- Question sur un délai, une échéance, un préavis, une période d'essai → calcul_delai_juridique
- Question sur un article de loi précis ou une référence du Code du travail → recherche_legifrance
- Question sur une actualité juridique récente, une jurisprudence, le SMIC actuel → web_search_juridique
- Salutation ou question générale → réponds directement sans outil

Utilise ce format exact :

Question : la question de l'utilisateur
Réflexion : réfléchis à quel outil utiliser et pourquoi
Action : nom de l'outil (parmi : {tool_names})
Entrée de l'action : l'input à passer à l'outil
Observation : le résultat de l'outil
... (répète si nécessaire)
Réflexion : je connais maintenant la réponse finale
Réponse finale : ta réponse complète, claire et professionnelle

Commence !

Question : {input}
Réflexion : {agent_scratchpad}"""

REACT_PROMPT = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)


# ===========================================================
# Construction de l'agent
# ===========================================================

def build_agent() -> AgentExecutor:
    print("[INFO] Chargement du LLM...")
    llm = load_llm()

    print("[INFO] Chargement de l'index FAISS (RAG)...")
    rag_tool = load_rag_tool(llm)

    tools = [rag_tool] + ALL_TOOLS

    print("[INFO] Création de l'agent ReAct...")
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=REACT_PROMPT,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )

    print("[SUCCESS] Agent juridique prêt.")
    return agent_executor


# ===========================================================
# Test CLI
# ===========================================================

if __name__ == "__main__":
    agent = build_agent()

    print("\n" + "=" * 60)
    print("Assistant Juridique Intelligent — Mode Terminal")
    print("Tapez 'quitter' pour arrêter.")
    print("=" * 60 + "\n")

    exemples = [
        "Quels sont mes droits en cas de licenciement ?",
        "Mon préavis commence le 15/04/2025, quelle est la date de fin pour 2 mois ?",
        "Quelles sont les règles sur les congés payés dans le Code du travail ?",
        "Quelles sont les obligations d'un entrepreneur de spectacles vivants ?",
        "Quelle est la jurisprudence récente sur le télétravail ?",
    ]
    print("Exemples de questions :")
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

        try:
            response = agent.invoke({"input": user_input})
            print(f"\nAssistant : {response['output']}\n")
        except Exception as e:
            print(f"[ERREUR] {e}\n")
