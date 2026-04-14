# Assistant Juridique Intelligent — RAG + Agents

Assistant conversationnel spécialisé en **droit du travail français**, combinant RAG, agents LangChain et mémoire conversationnelle.

## Fonctionnalités

- 📄 **RAG** : réponses basées sur des documents juridiques internes (FAISS)
- 🔧 **Agents & outils** : calcul de délais, recherche Légifrance, recherche web (Tavily)
- 🧠 **Mémoire conversationnelle** : suivi du contexte entre les messages
- 🔀 **Routage intelligent** : RAG / Agent / Conversation selon la question
- 💬 **Interface Chainlit** : chat interactif dans le navigateur

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| LLM | Groq — Llama 3.3-70b |
| RAG | LangChain + FAISS |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| Interface | Chainlit |
| Recherche web | Tavily |
| Source légale | Légifrance API |

## ⚙️ Installation

```bash
# Cloner le repo
git clone https://github.com/hajarharesrafia-sketch/GenerativeAiProject.git
cd GenerativeAiProject

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## 🔑 Configuration

Crée un fichier `.env` à la racine :

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
LEGIFRANCE_CLIENT_ID=your_client_id         # optionnel
LEGIFRANCE_CLIENT_SECRET=your_client_secret # optionnel
```

## ▶️ Lancement

```bash
chainlit run chainlit_app_v4.py -w
```

L'interface est disponible sur `http://localhost:8000`

