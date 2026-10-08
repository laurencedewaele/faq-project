---
title: App FAQ - Interface de recherche
description: Application Gradio de test de recherche dans la FAQ Guild Open Tech, qui interroge l'API FAQ et trace les échanges avec Langfuse
sdk: gradio
---

# App FAQ - Interface de recherche

Application Gradio permettant de tester la recherche dans la FAQ de Guild Open Tech. Elle interroge l'API FastAPI (`faq-backend-api`) déployée sur Hugging Face Spaces, affiche la réponse et les formulations candidates, et trace chaque échange dans Langfuse (avec recueil du feedback utilisateur).

## Caractéristiques

- 💬 **Interface de chat** : saisie d'une question, historique des échanges dans un `Chatbot`
- 🎛️ **Seuils réglables** : similarité, Top K, confiance et écart relatif du cross-encoder, modifiables à chaque requête
- 🔎 **Formulations candidates** : affichage des candidats avec leurs scores (similarité, cross-encoder) classés par score cross-encoder
- 📃 **Contenu de la FAQ** : liste complète de la FAQ consultable dans un accordéon (via `/list` de l'API)
- 👍👎 **Feedback** : retour utilisateur (OK / KO + commentaire) enregistré comme score Langfuse
- 📈 **Observabilité** : traces, métadonnées et scores envoyés à Langfuse

## Architecture

```
app.py                  # Application Gradio (UI, pipeline, appels API, Langfuse, feedback)
├── models.py           # Modèles Pydantic (QuestionInput, AnswerOutput, ConfigResponse, ...)
├── config.py           # Configuration (seuils, URL de l'API, lien Langfuse)
└── requirements.txt    # Dépendances Python
```

Flux d'une requête :

1. L'utilisateur saisit une question et valide
2. `faq_pipeline` ouvre une observation Langfuse (`faq-pipeline`) avec un `session_id` unique
3. `get_answer` envoie un `POST /ask` à l'API avec la question et les seuils
4. La réponse est convertie en `AnswerOutput`, les candidats sont triés et l'écart relatif (`gap`) entre les deux meilleurs est calculé
5. La réponse, les scores et les métadonnées sont enregistrés dans Langfuse, puis affichés dans l'interface

## Installation locale

### Prérequis
- Python 3.12
- pip ou conda
- Un compte Langfuse (clés API)

### Étapes

1. **Cloner le repository**
```bash
git clone <url>
cd faq-frontend
```

2. **Créer un environnement virtuel** (recommandé)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configurer les variables d'environnement** : créer un fichier `.env` (voir [Configuration](#configuration))

5. **Lancer l'application**
```bash
python app.py
```

L'application sera accessible à `http://localhost:7860`

## Utilisation

1. (Optionnel) Ouvrir l'accordéon **📃 Contenu de la FAQ** pour consulter les questions disponibles
2. Saisir une question dans le champ **Query** et valider avec Entrée
3. Lire la réponse dans le chat ; les scores (Similarity, Cross-Encoder, Gap) sont ajoutés à la réponse
4. Consulter la **Liste des formulations candidates** (similarité, score cross-encoder, formulation et thème)
5. Donner son avis avec **👍 Correct** / **👎 Uncorrect** (commentaire optionnel)
6. **Clear** réinitialise la conversation et remet les seuils à leurs valeurs par défaut

### Paramètres

| Paramètre | Description | Défaut |
|---|---|---|
| Similarity Threshold | Score minimal de similarité pour retenir un passage (1 = correspondance parfaite) | `0.85` |
| Top K | Nombre de candidats considérés | `3` |
| Confidence Threshold | Score cross-encoder minimal pour qu'un candidat soit jugé fiable | `0.60` |
| Relative Gap Threshold | Écart relatif minimal entre les deux meilleurs candidats pour valider une correspondance | `0.10` |

## Configuration

### Variables d'environnement

Créer un fichier `.env` :

```env
# URL de l'API FAQ (backend)
API_URL=https://loren-faq-backend.hf.space

# Seuils par défaut
SIMILARITY_THRESHOLD=0.85
CROSS_ENCODEUR_GAP_THRESHOLD=0.10
CROSS_ENCODEUR_CONFIDENCE_THRESHOLD=0.60

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Les valeurs par défaut sont définies dans `config.py`. Le nombre de candidats par défaut (`TOP_K = 3`) et le lien vers les traces Langfuse (`LANGFUSE_LINK`) y sont également définis.

## API consommée

L'application dépend de l'API FAQ (`API_URL`) :

| Endpoint | Usage |
|---|---|
| `GET /config` | Récupère les noms des modèles (embeddings, cross-encoder) au chargement de la page |
| `GET /list` | Récupère le contenu de la FAQ affiché dans l'accordéon |
| `POST /ask` | Envoie la question et les seuils, retourne la réponse et les candidats |

Le contenu de la FAQ est chargé au démarrage de l'application : l'API doit donc être disponible (le Space peut mettre quelques dizaines de secondes à se réveiller après une période d'inactivité).

## Langfuse

Chaque question génère une trace `faq-search` (tag `FAQ`) contenant :

- **Entrée / sortie** : question et réponse
- **Statut** : `SUCCESS` si la confiance est atteinte, `NO_MATCH` sinon, `ERROR` en cas d'échec
- **Métadonnées** : formulation, thème, seuils, candidats, URL de l'API, modèles utilisés
- **Scores** : `similarity`, `cross_encoder_logit`, `cross_encoder_score` (sigmoïde du logit) et `gap`
- **Feedback utilisateur** : score `user_feedback` (`OK` / `KO`) rattaché à la trace avec le commentaire éventuel

Si Langfuse n'est pas configuré, l'initialisation du client est signalée dans les logs.

## Hugging Face

L'application est déployée sur Hugging Face Spaces (SDK Gradio 5.49.1, Python 3.12, point d'entrée `app.py`).

Dans l'interface du Space, "Settings" → "Variables and secrets" permet de définir les variables d'environnement (clés Langfuse, `API_URL`, seuils). Les logs de build et d'exécution sont consultables depuis l'onglet "Logs".

## Logs

Les logs sont affichés au démarrage et en temps réel :
- 🤖 Démarrage et configuration (URL de l'API, état de Langfuse)
- 🔍 Questions envoyées à l'API
- ✅ Réponses récupérées et feedbacks enregistrés
- ⚠️/❌ Avertissements et erreurs (API, Langfuse)
