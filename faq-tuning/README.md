# got-faq-tuning

Recherche d'hyperparamètres pour l'endpoint `POST /ask` de `got-faq-api`, via
[Optuna](https://optuna.org/) pour l'optimisation et [MLflow](https://mlflow.org/)
pour le tracking des expériences.

## Principe

Le script `tune.py` :

1. Charge la FAQ (`got-faq-api/data/*.json`) et instancie `EmbeddingManager`
   **exactement comme le fait `got-faq-api` au démarrage** (mêmes classes,
   importées directement depuis `../got-faq-api`).
2. Pour chaque essai Optuna, construit un `QuestionInput` avec une
   configuration d'hyperparamètres et appelle `EmbeddingManager.search_similar_faq`
   — la fonction réellement utilisée par `POST /ask` — sur chaque échantillon
   du CSV.
3. Compare la réponse obtenue à l'`intent_key` attendu et calcule le taux de
   bonnes réponses (`accuracy_theme`).
4. Trace chaque essai (paramètres + métriques) dans MLflow, puis à la fin de
   l'étude, log les graphiques Optuna (historique d'optimisation, importances
   des paramètres, coordonnées parallèles, slice) comme artefacts MLflow.

Les modèles d'embeddings et de cross-encoder ne sont chargés **qu'une seule
fois** au démarrage du script (pas à chaque essai), et un cache d'inférence
mémorise les résultats d'encodage par question déjà vue, pour accélérer la
recherche sur de nombreux essais.

## Hyperparamètres optimisés

Ce sont les 4 paramètres déjà exposés par `QuestionInput` (aucune modification
de `got-faq-api` n'est nécessaire) :

| Paramètre | Rôle |
|---|---|
| `similarity_threshold` | Seuil de similarité cosinus minimal pour retenir un candidat |
| `top_k` | Nombre de thèmes distincts considérés |
| `cross_encodeur_gap_threshold` | Écart relatif minimal entre le 1er et le 2e candidat re-classés |
| `cross_encodeur_confidence_threshold` | Score cross-encoder minimal pour répondre avec confiance |

## Installation

```bash
cd got-faq-tuning
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Jeu de données

`data/dataset.csv` contient les utterances annotées (colonnes
`utterance,intent_key,category`).

- `utterance` : une formulation utilisateur à évaluer.
- `intent_key` : l'intention attendue, correspondant au thème du JSON FAQ.
- `category` : la catégorie métier conservée dans les détails du tuning.

## Lancer le tuning

```bash
python tune.py --n-trials 50
```

Options utiles :

```bash
python tune.py \
   --dataset data/dataset.csv \
  --n-trials 100 \
  --study-name faq-ask-tuning-v2 \
  --mlflow-tracking-uri ./mlruns \
  --similarity-threshold-range 0.3 0.9 \
  --top-k-range 1 5 \
  --gap-threshold-range 0.0 0.5 \
  --confidence-threshold-range 0.3 0.95
```

## Consulter les résultats

```bash
mlflow ui --backend-store-uri ./mlruns
```

Puis ouvrir http://localhost:5000 : chaque essai apparaît comme un run
imbriqué sous le run d'étude, avec ses paramètres et son `accuracy_theme`. Les graphiques Optuna (HTML interactifs) sont
disponibles dans les artefacts du run parent, sous `optuna_plots/`.

## Évaluer une configuration avec LangSmith

`langsmith_evaluate.py` teste **une seule** configuration d'hyperparamètres
(pas de recherche) et trace le résultat dans LangSmith : un dataset
`got-faq-dataset`, un projet `got-faq-evaluate`, et une expérience par
exécution.

```bash
copy .env.example .env      # puis renseigner LANGSMITH_API_KEY
python langsmith_evaluate.py --similarity-threshold 0.85 --top-k 3 --gap-threshold 0.1 --confidence-threshold 0.6
```

Sans argument, les valeurs par défaut sont celles de `got-faq-api/config.py`.

Le fichier `.env` (voir `.env.example`) contient la clé `LANGSMITH_API_KEY` et,
si le compte est hébergé en Europe, `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`
(à commenter pour la région US).

Autres options :

| Option | Rôle |
|---|---|
| `--dataset` | CSV source (défaut : `data/dataset.csv`) |
| `--faq-json` | JSON de FAQ à charger (défaut : celui de `got-faq-api`) |
| `--dataset-name` | Nom du dataset LangSmith (défaut : `got-faq-dataset`) |
| `--project` | Projet de tracing LangSmith (défaut : `got-faq-evaluate`) |
| `--experiment-prefix` | Préfixe du nom d'expérience (défaut : nom du projet) |
| `--recreate-dataset` | Supprime et réimporte le dataset depuis le CSV |
| `--max-concurrency` | Nombre d'exemples évalués en parallèle (défaut : 1) |
| `--num-repetitions` | Nombre de répétitions de chaque exemple (défaut : 1) |

Le dataset LangSmith est créé au premier lancement depuis `data/dataset.csv`
(entrée : la question ; sortie attendue : `expected_theme` et `category`) puis
réutilisé tel quel, pour que toutes les expériences portent sur les mêmes
exemples. Utiliser `--recreate-dataset` après une modification du CSV.

Métriques calculées :

| Métrique | Sens |
|---|---|
| `theme_correct` | Réponse donnée avec confiance ET sur le bon thème |
| `answered` | L'API a répondu au lieu de s'abstenir |
| `false_positive` | Réponse confiante mais sur le mauvais thème |
| `accuracy_theme` / `answer_rate` / `precision_when_answered` | Agrégats sur l'ensemble du dataset |

La configuration testée et les noms des deux modèles sont enregistrés dans les
métadonnées de l'expérience : c'est ce qui permet de comparer deux exécutions
dans l'onglet *Experiments* du dataset. Le détail par exemple est aussi exporté
dans `artifacts/langsmith/<nom_experience>.csv`.
