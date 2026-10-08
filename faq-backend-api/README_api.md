---
title: API FAQ - Embeddings Multilingues
description: API FastAPI pour répondre à des questions via recherche de similarité sur une base FAQ avec embeddings multilingues
sdk: docker
---

# API FAQ - Embeddings Multilingues

API FastAPI permettant de répondre à des questions en cherchant la réponse la plus similaire dans une base FAQ via recherche d'embeddings multilingues.

## Caractéristiques

- 🌍 **Multilingue** : Utilise le modèle `intfloat/multilingual-e5-small` pour supporter plusieurs langues
- 🚀 **Rapide** : Recherche par similarité cosinus via ChromaDB
- 📊 **Seuil configurable** : Filtrage des réponses selon un seuil de confiance
- 📝 **Type-safe** : Utilise Pydantic pour la validation des données
- 📚 **Documentation interactive** : Swagger UI intégrée via `/docs`

## Architecture

```
app.py                  # Point d'entrée FastAPI
├── models.py           # Modèles Pydantic (QuestionInput, AnswerOutput)
├── config.py           # Configuration centralisée
├── faq_loader.py       # Chargement de la FAQ au format JSON
├── embeddings.py       # Gestion ChromaDB + modèle transformers
└── data/
    └── faq.json         # Exemple de base FAQ (intentions, formulations, réponses)
```

## Installation locale

### Prérequis
- Python 3.8+
- pip ou conda

### Étapes

1. **Cloner le repository** (ou télécharger les fichiers)
```bash
git clone <url>
cd faq-backend-api
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

4. **Configurer les variables d'environnement** (optionnel)
```bash
cp .env.example .env
# Éditer .env si nécessaire
```

5. **Lancer l'application**
```bash
uvicorn app:app --reload
```

L'API sera accessible à `http://localhost:8000`

## Utilisation

### Endpoint principal : POST `/ask`

Poser une question et obtenir la réponse la plus similaire

**Requête :**
```json
{
  "question": "Est-ce que c’est gratuit ?"
}
```

**Réponse :**
```json
{
  "question": "Est-ce que c’est gratuit ?",
  "formulation": "Est-ce que c’est gratuit ?",
  "answer": "Oui. L’adhésion et la participation aux projets de l'association sont gratuites. Il n’y a ni frais de formation ni cotisation obligatoire pour contribuer aux projets.",
  "theme": "est_ce_que_c_est_gratuit",
  "similarity_score": 0.95,
  "confidence": true
}
```

**Champs de réponse :**
- `question` : Question posée
- `formulation` : Formulation de la FAQ qui correspond le mieux à la question
- `answer` : Réponse trouvée dans la FAQ
- `theme` : Intention de la FAQ (`intent_key`)
- `similarity_score` : Score de similarité cosinus [0, 1]
- `confidence` : `true` si le score ≥ seuil de confiance, `false` sinon

### Endpoint santé : GET `/health`

Vérifier l'état du service et le nombre de FAQs indexées

**Réponse :**
```json
{
  "status": "ok",
  "faq_count": 474
}
```

### Documentation interactive : GET `/docs`

Accéder à la documentation Swagger UI interactive :
```
http://localhost:8000/docs
```

## Tests

### Avec curl

```bash
# Test question connue
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Est-ce que c’est gratuit ?"}'

# Test question inconnue
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Comment faire des crêpes ?"}'

# Health check
curl "http://localhost:8000/health"
```

### Avec Python

```python
import requests

url = "http://localhost:8000/ask"
response = requests.post(url, json={"question": "Est-ce que c’est gratuit ?"})
print(response.json())
```

### Avec Postman ou Insomnia

Importer la collection depuis `/docs` ou créer manuellement une requête POST vers `/ask`

## Configuration

### Variables d'environnement

Créer un fichier `.env` (voir `.env.example`) pour configurer :

```env
# Modèle d'embedding
MODEL_NAME=intfloat/multilingual-e5-small

# Seuil de confiance [0.0, 1.0]
# 0.5 par défaut
SIMILARITY_THRESHOLD=0.5
```

### Fichiers de données

Le fichier d'exemple `data/faq.json` illustre le format attendu (chemin défini par `FAQ_JSON_PATH` dans `config.py`). Il contient une liste `intents` ; chaque intent activé (`enabled: true`) comporte :
- `intent_key` : identifiant de l'intention (renvoyé comme `theme`)
- `canonical_question` et `utterances` : formulations de la question
- `response` : réponse associée

Les FAQs sont chargées et indexées au démarrage de l'application.

## Hugging Face

L'API est déployée sur Hugging Face Spaces (runtime Docker) : **https://loren-faq-backend-api.hf.space**

| Ressource | URL |
|---|---|
| Swagger UI | https://loren-faq-backend-api.hf.space/docs |
| Question (POST) | https://loren-faq-backend-api.hf.space/ask |
| Santé | https://loren-faq-backend-api.hf.space/health |
| Configuration | https://loren-faq-backend-api.hf.space/config |

Exemple :

```bash
curl -X POST "https://loren-faq-backend-api.hf.space/ask"   -H "Content-Type: application/json"   -d '{"question": "Comment je deviens membre ?"}'
```

### Configuration

Dans l'interface du Space, "Settings" → "Variables and secrets" permet de définir les variables d'environnement (ex : `SIMILARITY_THRESHOLD=0.6`). Les logs de build et d'exécution sont consultables depuis l'onglet "Logs".

### Démarrage

Le démarrage (~30-60s) comprend :
1. Installation des dépendances
2. Téléchargement des modèles (embeddings et cross-encoder)
3. Chargement de la FAQ
4. Génération des embeddings

Après une période d'inactivité, le Space peut être mis en veille : la première requête peut alors prendre quelques dizaines de secondes.

## Architecture ChromaDB

### EphemeralClient (Option utilisée)

Les embeddings sont chargés en mémoire au démarrage et recalculés à chaque redémarrage :

**Avantages :**
- ✓ Plus rapide (pas d'I/O disque)
- ✓ Pas de dépendance de persistance
- ✓ Idéal pour HF Spaces (ressources limitées)

**Inconvénients :**
- Les embeddings sont recalculés à chaque déploiement (~30s)

### PersistentClient (Alternative)

Si vous préférez persister les embeddings sur disque, remplacer `EphemeralClient` par `PersistentClient` dans `embeddings.py`.

## Considérations de performance

- **Modèle** : `multilingual-e5-small` (~50MB)
- **Threshold** : Augmenter pour être plus strict, réduire pour plus de matches
- **Top-k** : 3 par défaut (`TOP_K` dans `config.py`) : nombre maximum de thèmes distincts retenus comme candidats (`list_candidates`). Peut être surchargé par requête via le champ `top_k` de `/ask`

## Logs

Les logs sont affichés au démarrage et en temps réel :
- 🚀 Démarrage
- 📚 Chargement FAQ
- ✓ Application prête
- 📝 Questions reçues
- ✓/❌ Réponses trouvées
