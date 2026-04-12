"""
tools.py — Outils pour l'agent juridique (Partie 2)
Outils : Recherche web (Tavily), Calcul de délais juridiques, API Légifrance (mode démo)
"""

from __future__ import annotations

import os
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from langchain.tools import tool


# ===========================================================
# OUTIL 1 : Recherche web juridique (Tavily)
# ===========================================================

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


@tool
def web_search_juridique(query: str) -> str:
    """
    Effectue une recherche web pour trouver des informations juridiques récentes :
    actualités légales, jurisprudences, nouvelles lois, décrets, articles de droit.
    Utilise cet outil quand la question porte sur des informations qui ne sont
    pas dans les documents internes ou qui nécessitent une mise à jour récente.
    Exemples : 'jurisprudence licenciement abusif 2024', 'nouveau SMIC 2025'.
    Input : une requête de recherche en français.
    """
    if not TAVILY_API_KEY:
        return (
            "[Mode démo] Recherche web désactivée. "
            "Ajoutez TAVILY_API_KEY dans votre .env pour activer cet outil.\n"
            f"Requête tentée : '{query}'\n"
            "Résultat simulé : Pour cette question, consultez le site service-public.fr "
            "ou legifrance.gouv.fr pour des informations à jour."
        )

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": 3,
        "include_answer": True,
        "include_domains": [
            "legifrance.gouv.fr",
            "service-public.fr",
            "travail-emploi.gouv.fr",
            "juritravail.com",
            "droit-travail-france.fr",
        ],
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("answer"):
            return f"Résultat : {data['answer']}"

        results = data.get("results", [])
        if not results:
            return "Aucun résultat trouvé pour cette requête juridique."

        summaries = []
        for r in results[:3]:
            title = r.get("title", "Sans titre")
            content = r.get("content", "")[:400]
            url_source = r.get("url", "")
            summaries.append(f"• {title}\n  {content}\n  Source : {url_source}")

        return "\n\n".join(summaries)

    except Exception as e:
        return f"Erreur lors de la recherche web : {e}"


# ===========================================================
# OUTIL 2 : Calcul de délais juridiques
# ===========================================================

@tool
def calcul_delai_juridique(description: str) -> str:
    """
    Calcule des délais juridiques à partir d'une date de départ et d'une durée.
    Gère les délais en jours, semaines, mois ou années.
    Utilise cet outil pour toute question sur des échéances légales :
    préavis, délais de recours, périodes d'essai, ancienneté, congés, etc.

    Format d'input attendu (en langage naturel) :
    - 'préavis de 2 mois à partir du 15/04/2025'
    - 'délai de recours de 2 mois depuis le 01/03/2025'
    - 'période d essai de 3 mois à partir du 10/01/2025'
    - 'ancienneté de 3 ans depuis le 05/06/2022'
    - 'délai de 30 jours à partir du 20/04/2025'
    """
    import re

    texte = description.lower()

    # --- Extraction de la date de départ ---
    date_pattern = r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})'
    date_match = re.search(date_pattern, texte)

    if not date_match:
        return (
            "❌ Impossible de trouver une date dans votre demande.\n"
            "Veuillez préciser une date au format JJ/MM/AAAA.\n"
            "Exemple : 'préavis de 2 mois à partir du 15/04/2025'"
        )

    jour, mois, annee = date_match.groups()
    annee = int(annee)
    if annee < 100:
        annee += 2000

    try:
        date_debut = datetime(annee, int(mois), int(jour))
    except ValueError:
        return "❌ Date invalide. Vérifiez le format JJ/MM/AAAA."

    # --- Extraction de la durée et unité ---
    duree_match = re.search(r'(\d+)\s*(jour|jours|semaine|semaines|mois|an|ans|année|années)', texte)

    if not duree_match:
        return (
            "❌ Impossible de trouver la durée dans votre demande.\n"
            "Précisez une durée en jours, semaines, mois ou ans.\n"
            "Exemple : 'délai de 30 jours à partir du 20/04/2025'"
        )

    quantite = int(duree_match.group(1))
    unite = duree_match.group(2)

    # --- Calcul de la date de fin ---
    if "jour" in unite:
        date_fin = date_debut + timedelta(days=quantite)
        duree_label = f"{quantite} jour(s)"
    elif "semaine" in unite:
        date_fin = date_debut + timedelta(weeks=quantite)
        duree_label = f"{quantite} semaine(s)"
    elif "mois" in unite:
        date_fin = date_debut + relativedelta(months=quantite)
        duree_label = f"{quantite} mois"
    else:  # ans / années
        date_fin = date_debut + relativedelta(years=quantite)
        duree_label = f"{quantite} an(s)"

    # --- Résultat ---
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
# OUTIL 3 : Recherche dans Légifrance (mode démo)
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
            "et de 2 mois pour une ancienneté supérieure à 2 ans. "
            "Des dispositions conventionnelles peuvent prévoir des durées plus longues."
        ),
        "url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006901248",
    },
    "congés payés": {
        "titre": "Article L3141-3 — Code du travail",
        "resume": (
            "Le salarié a droit à un congé de 2,5 jours ouvrables par mois de travail "
            "effectif chez le même employeur, soit 30 jours ouvrables (5 semaines) par an."
        ),
        "url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000033020420",
    },
    "salaire": {
        "titre": "Article L3231-2 — Code du travail (SMIC)",
        "resume": (
            "Tout salarié perçoit une rémunération au moins égale au SMIC. "
            "Le SMIC est revalorisé au minimum chaque année au 1er janvier. "
            "Il peut également être revalorisé en cours d'année si l'inflation dépasse 2%."
        ),
        "url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006902626",
    },
    "période d'essai": {
        "titre": "Article L1221-19 — Code du travail",
        "resume": (
            "La période d'essai est de 2 mois pour les ouvriers et employés, "
            "3 mois pour les agents de maîtrise et techniciens, "
            "4 mois pour les cadres. Elle peut être renouvelée une fois si un accord de branche le prévoit."
        ),
        "url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000019071188",
    },
}


@tool
def recherche_legifrance(query: str) -> str:
    """
    Recherche des articles de loi et textes juridiques officiels dans Légifrance.
    Utilise cet outil pour trouver les références légales précises du Code du travail
    ou d'autres textes de loi français (articles, décrets, jurisprudences officielles).
    Exemples : 'article sur le licenciement', 'règles sur les congés payés',
    'durée période d essai cadre', 'calcul préavis'.
    Input : une requête juridique en français.
    """
    LEGIFRANCE_CLIENT_ID = os.getenv("LEGIFRANCE_CLIENT_ID", "")
    LEGIFRANCE_CLIENT_SECRET = os.getenv("LEGIFRANCE_CLIENT_SECRET", "")

    # --- Mode réel (si credentials disponibles) ---
    if LEGIFRANCE_CLIENT_ID and LEGIFRANCE_CLIENT_SECRET:
        try:
            token_response = requests.post(
                "https://oauth.piste.gouv.fr/api/oauth/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": LEGIFRANCE_CLIENT_ID,
                    "client_secret": LEGIFRANCE_CLIENT_SECRET,
                    "scope": "openid",
                },
                timeout=10,
            )
            token_response.raise_for_status()
            access_token = token_response.json()["access_token"]

            search_response = requests.post(
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
                        "typePagination": "DEFAUT",
                    },
                    "fond": "CODE_DATE",
                },
                timeout=10,
            )
            search_response.raise_for_status()
            data = search_response.json()

            results = data.get("results", [])
            if not results:
                return f"Aucun article trouvé dans Légifrance pour : '{query}'."

            output = [f"📚 Résultats Légifrance pour : '{query}'\n"]
            for r in results[:3]:
                titre = r.get("titre", "Sans titre")
                texte = r.get("extract", "")[:400]
                lien = r.get("cid", "")
                output.append(f"• {titre}\n  {texte}\n  🔗 legifrance.gouv.fr/codes/id/{lien}")

            return "\n\n".join(output)

        except Exception as e:
            return f"Erreur API Légifrance : {e}"

    # --- Mode démo ---
    query_lower = query.lower()
    for keyword, data in LEGIFRANCE_DEMO_DATA.items():
        if keyword in query_lower:
            return (
                f"📚 [Mode démo] Résultat Légifrance\n"
                f"──────────────────────────────────\n"
                f"📄 {data['titre']}\n\n"
                f"{data['resume']}\n\n"
                f"🔗 Source officielle : {data['url']}\n\n"
                f"⚠️ Mode démonstration — Ajoutez LEGIFRANCE_CLIENT_ID et "
                f"LEGIFRANCE_CLIENT_SECRET dans votre .env pour des résultats réels."
            )

    return (
        f"📚 [Mode démo] Aucune donnée simulée pour '{query}'.\n"
        f"En mode réel, cet outil interrogerait l'API officielle Légifrance.\n"
        f"Thèmes disponibles en démo : licenciement, préavis, congés payés, salaire, période d'essai.\n"
        f"🔗 Consultez directement : https://www.legifrance.gouv.fr"
    )


# Liste exportée pour l'agent
ALL_TOOLS = [web_search_juridique, calcul_delai_juridique, recherche_legifrance]
