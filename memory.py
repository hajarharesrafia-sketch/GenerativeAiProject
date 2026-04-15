"""
memory.py — Partie 4 : Mémoire conversationnelle
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gère l'historique des échanges et le formate pour
l'injecter dans chaque appel LLM.

Utilisation :
    from memory import ConversationMemory
    memory = ConversationMemory(max_turns=10)
    memory.add("Bonjour", "Bonjour ! Comment puis-je vous aider ?")
    history_text = memory.get_history_as_text()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class Turn:
    """Un échange (question + réponse) dans la conversation."""
    user: str
    assistant: str


class ConversationMemory:
    """
    Mémoire conversationnelle simple basée sur une liste de tours.

    Paramètres :
        max_turns : nombre maximum de tours conservés en mémoire.
                    Les plus anciens sont supprimés en premier (fenêtre glissante).
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._history: List[Turn] = []

    # ───────────────────────────────────────────
    # API publique
    # ───────────────────────────────────────────

    def add(self, user_message: str, assistant_message: str) -> None:
        """Ajoute un tour à l'historique. Supprime le plus ancien si nécessaire."""
        self._history.append(Turn(user=user_message, assistant=assistant_message))
        if len(self._history) > self.max_turns:
            self._history.pop(0)

    def get_history_as_text(self) -> str:
        """
        Retourne l'historique formaté en texte brut pour l'injection dans un prompt.

        Exemple de sortie :
            Utilisateur : Bonjour
            Assistant : Bonjour ! Comment puis-je vous aider ?

            Utilisateur : Quels sont mes droits ?
            Assistant : Selon le document...
        """
        if not self._history:
            return ""

        lines = []
        for turn in self._history:
            lines.append(f"Utilisateur : {turn.user}")
            lines.append(f"Assistant : {turn.assistant}")
            lines.append("")          # ligne vide entre les tours
        return "\n".join(lines).strip()

    def get_history_as_messages(self) -> list:
        """
        Retourne l'historique sous forme de liste de messages LangChain
        (HumanMessage / AIMessage) pour injection directe dans un LLM.
        """
        from langchain_core.messages import HumanMessage, AIMessage
        messages = []
        for turn in self._history:
            messages.append(HumanMessage(content=turn.user))
            messages.append(AIMessage(content=turn.assistant))
        return messages

    def clear(self) -> None:
        """Efface complètement l'historique."""
        self._history = []

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"ConversationMemory(turns={len(self._history)}/{self.max_turns})"
