"""
tools.py — Outils pour l'agent juridique (Partie 2)
Outils : Recherche web (Tavily), Calcul de délais juridiques, Légifrance (mode démo)
Style inspiré du notebook du professeur
"""

from __future__ import annotations

import os
import re
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from langchain.tools import Tool


# ===========================================================
# OUTIL 1 : Recherche web juridique (Tavily)
# ===========================================================

def web_search_juridique(query: str) -> str:
    """Recherche web pour trouver des informations juridiques récentes."""
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return (
            "[Mode démo] Recherche web désactivée. "
            f"Ajoutez TAVILY_API_KEY dans votre .env.\n"
            f"Requête tentée : '{query}'"
        )

    try:
        client = TavilyClient(api_key=api_key)
        res = client.search(query=query, search_depth="basic", max_results=3)
        results = res.get("results", [])

        if not results:
            return "Aucun résultat trouvé."

        lines = []
        for r in results[:3]:
            title = r.get("title", "Sans titre")
            content = r.get("content", "")[:400]
            url = r.get("url", "")
            lines.append(f"- {title}\n  {content}\n  {url}")

        return "\n\n".join(lines)

    except Exception as e:
        return f"Erreur lors de la recherche web : {e}"


# ===========================================================
# OUTIL 2 : Calcul de délais juridiques
# ===========================================================

def calcul_delai_juridique(description: str) -> str:
    """Calcule des délais juridiques à partir d'une date et d'une durée."""
    texte = description.lower()

    date_pattern = r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})'
    date_match = re.search(date_pattern, texte)

    if not date_match:
        return (
            "❌ Impossible de trouver une date. "
            "Format attendu : 'préavis de 2 mois à partir du 15/04/2025'"
        )

    jour, mois, annee = date_match.groups()
    annee = int(annee)
    if annee < 100:
        annee += 2000

    try:
        date_debut = datetime(annee, int(mois), int(jour))
    except ValueError:
        return "❌ Date invalide. Vérifiez le format JJ/MM/AAAA."

    duree_match = re.search(
        r'(\d+)\s*(jour|jours|semaine|semaines|mois|an|ans|année|années)', texte
    )

    if not duree_match:
        return (
            "❌ Impossible de trouver la durée. "
            "Exemple : 'délai de 30 jours à partir du 20/04/2025'"
        )

    quantite = int(duree_match.group(1))
    unite = duree_match.group(2)

    if "jour" in unite:
        date_fin = date_debut + timedelta(days=quantite)
        duree_label = f"{quantite} jour(s)"
    elif "semaine" in unite:
        date_fin = date_debut + timedelta(weeks=quantite)
        duree_label = f"{quantite} semaine(s)"
    elif "mois" in unite:
        date_fin = date_debut + relativedelta(months=quantite)
        duree_label = f"{quantite} mois"
    else:
        date_fin = date_debut + relativedelta(years=quantite)
        duree_label = f"{quantite} an(s)"

    jours_restants = (date_fin - datetime.now()).days

    if jours_restants > 0:
        statut = f"⏳ Il reste {jours_restants} jour(s) avant l'échéance."
    elif jours_restants == 0:
        statut = "⚠️ L'échéance est aujourd'hui !"
    else:
        statut = f"❗ Cette échéance est dépassée depuis {abs(jours_restants)} jour(s)."

    return (
        f"📅 Calcul du délai juridique\n"
        f"──────────────────────────\n"
        f"Date de départ : {date_debut.strftime('%d/%m/%Y')}\n"
        f"Durée : {duree_label}\n"
        f"Date d'échéance : {date_fin.strftime('%d/%m/%Y')}\n"
        f"{statut}"
    )


# ===========================================================
# OUTIL 3 : Recherche Légifrance (mode démo)
# ===========================================================

LEGIFRANCE_DEMO_DATA = {
    "licenciement": {
        "titre": "Article L1237-19 — Code du travail",
        "resume": (
            "La rupture conventionnelle permet à l'employeur et au salarié de convenir "
            "d'un commun accord des conditions de rupture du contrat de travail. "
            "Elle ouvre droit au salarié à l'allocation chômage."
        ),
        "url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000019068011",
    },
    "préavis": {
        "titre": "Article L1234-1 — Code du travail",
        "resume": (
            "Le préavis est de 1 mois pour une ancienneté entre 6 mois et 2 ans, "
            "et de 2 mois pour une ancienneté supérieure à 2 ans."
        ),
        "url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006901248",
    },
    "conges": {
        "titre": "Article L3141-3 — Code du travail",
        "resume": (
            "Le salarié a droit à 2,5 jours ouvrables par mois de travail effectif, "
            "soit 30 jours ouvrables (5 semaines) par an."
        ),
        "url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000033020420",
    },
    "salaire": {
        "titre": "Article L3231-2 — Code du travail (SMIC)",
        "resume": (
            "Tout salarié perçoit une rémunération au moins égale au SMIC, "
            "revalorisé au minimum chaque année au 1er janvier."
        ),
        "url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006902626",
    },
    "periode d'essai": {
        "titre": "Article L1221-19 — Code du travail",
        "resume": (
            "La période d'essai est de 2 mois pour les ouvriers et employés, "
            "3 mois pour les agents de maîtrise, 4 mois pour les cadres."
        ),
        "url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000019071188",
    },
    "cddu": {
        "titre": "Article L1242-2 — Code du travail (CDDU)",
        "resume": (
            "Dans le spectacle, il est possible de conclure des contrats à durée déterminée "
            "dits d'usage pour certains emplois par nature temporaires."
        ),
        "url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006901208",
    },
}


def recherche_legifrance(query: str) -> str:
    """Recherche des articles de loi dans Légifrance."""
    client_id = os.getenv("LEGIFRANCE_CLIENT_ID", "")
    client_secret = os.getenv("LEGIFRANCE_CLIENT_SECRET", "")

    if client_id and client_secret:
        try:
            token_resp = requests.post(
                "https://oauth.piste.gouv.fr/api/oauth/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "scope": "openid",
                },
                timeout=10,
            )
            token_resp.raise_for_status()
            access_token = token_resp.json()["access_token"]

            search_resp = requests.post(
                "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app/search",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "recherche": {
                        "query": query,
                        "pageNumber": 1,
                        "pageSize": 3,
                        "sort": "PERTINENCE",
                    },
                    "fond": "CODE_DATE",
                },
                timeout=10,
            )
            search_resp.raise_for_status()
            results = search_resp.json().get("results", [])

            if not results:
                return f"Aucun article trouvé pour : '{query}'."

            output = []
            for r in results[:3]:
                titre = r.get("titre", "Sans titre")
                texte = r.get("extract", "")[:400]
                output.append(f"📄 {titre}\n{texte}")
            return "\n\n".join(output)

        except Exception as e:
            return f"Erreur API Légifrance : {e}"

    # Mode démo
    query_lower = query.lower()
    for keyword, data in LEGIFRANCE_DEMO_DATA.items():
        if keyword in query_lower:
            return (
                f"📚 [Mode démo] Résultat Légifrance\n"
                f"──────────────────────────────────\n"
                f"📄 {data['titre']}\n\n"
                f"{data['resume']}\n\n"
                f"🔗 {data['url']}\n\n"
                f"⚠️ Mode démo — Ajoutez LEGIFRANCE_CLIENT_ID et LEGIFRANCE_CLIENT_SECRET "
                f"dans votre .env pour des résultats réels."
            )

    return (
        f"📚 [Mode démo] Aucune donnée pour '{query}'.\n"
        f"Thèmes disponibles : licenciement, préavis, congés, salaire, "
        f"période d'essai, CDDU.\n"
        f"🔗 https://www.legifrance.gouv.fr"
    )


# ===========================================================
# Liste des outils au format LangChain (style du prof)
# ===========================================================

ALL_TOOLS = [
    Tool(
        name="web_search_juridique",
        func=web_search_juridique,
        description=(
            "Recherche web pour trouver des informations juridiques récentes : "
            "actualités légales, jurisprudences, nouvelles lois, SMIC actuel. "
            "Utilise cet outil quand la question nécessite des informations récentes "
            "ou qui ne sont pas dans les documents internes."
        ),
    ),
    Tool(
        name="calcul_delai_juridique",
        func=calcul_delai_juridique,
        description=(
            "Calcule des délais juridiques à partir d'une date et d'une durée. "
            "Utilise cet outil pour : préavis, périodes d'essai, délais de recours, "
            "ancienneté, congés. "
            "Exemple d'input : 'préavis de 2 mois à partir du 15/04/2025'"
        ),
    ),
    Tool(
        name="recherche_legifrance",
        func=recherche_legifrance,
        description=(
            "Recherche des articles de loi officiels dans Légifrance (Code du travail). "
            "Utilise cet outil pour trouver des références légales précises : "
            "articles sur le licenciement, les congés payés, le SMIC, la période d'essai, "
            "le CDDU dans le spectacle, etc."
        ),
    ),
]
