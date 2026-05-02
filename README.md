# 🌦️ Prédiction de Pluie en Australie (Rain Prediction)

Ce projet est une application complète de Machine Learning visant à prédire s'il pleuvra le lendemain (`RainTomorrow`) en Australie, en se basant sur un historique de données météorologiques. Il a été conçu de manière modulaire, orientée objet et met l'accent sur les bonnes pratiques de développement logiciel en Data Science (configuration centralisée, modèles encapsulés, validation croisée rigoureuse, et une interface utilisateur interactive).

## ✨ Fonctionnalités Principales

- **Architecture Modulaire Orientée Objet** : Chaque algorithme de machine learning hérite d'une classe de base abstraite `BaseModel`, encapsulant sa propre logique de prétraitement, d'entraînement, d'optimisation et d'évaluation.
- **Pipeline d'Entraînement Automatisé** : Un script `run.py` en ligne de commande pour préparer les données, entraîner, valider (Validation croisée) et évaluer différents modèles.
- **Optimisation des Hyperparamètres** : Intégration de `GridSearchCV` paramétrable à la volée via un fichier de configuration `param.yaml`.
- **Évaluation Rigoureuse** : Génération automatique de rapports d'expérimentation complets avec matrices de confusion, rapports de classification, ainsi que la sauvegarde de graphiques d'évaluation (courbes ROC et Precision-Recall).
- **Application Interactive (Dashboard)** : Une interface web Streamlit intuitive permettant d'effectuer des scénarios "What-If" (simulations en temps réel) en ajustant les paramètres météo via des curseurs et de comparer les performances des modèles entraînés.

## 📂 Structure du Projet

```text
rain_prediction/
├── app.py                     # Dashboard interactif (Streamlit)
├── run.py                     # Script principal d'entraînement et d'évaluation en CLI
├── param.yaml                 # Fichier de configuration central (hyperparamètres, features)
├── pyproject.toml / uv.lock   # Fichiers de gestion des dépendances
├── data/                      # Dossier contenant les jeux de données (bruts et traités)
├── notebooks/                 # Notebooks Jupyter pour l'analyse exploratoire (EDA)
├── models/                    # Scripts contenant l'architecture des modèles
│   ├── base_model.py          # Classe de base (BaseModel) définissant le contrat
│   ├── regression_logistique.py # Implémentation de la régression logistique
│   ├── mlp.py                 # Implémentation du réseau de neurones (MLP)
│   ├── sgd.py                 # Implémentation du SVM via SGD
│   └── model_registry.py      # Registre centralisant l'accès aux modèles
├── model_output/              # Sauvegarde des modèles entraînés (.pkl) et rapports (.txt, .png)
└── utils/                     # Scripts utilitaires divers (séparation des données, reporting...)
```

## 🛠️ Modèles Implémentés

Le projet permet d'entraîner et de comparer facilement plusieurs approches :
1. **Régression Logistique** (`logistic`) : Modèle linéaire simple, robuste et interprétable, souvent utilisé comme "baseline".
2. **Multi-Layer Perceptron** (`mlp`) : Réseau de neurones artificiel basique (`MLPClassifier`), capable de capter des relations non linéaires plus complexes.
3. **Stochastic Gradient Descent** (`sgd`) : Modèle optimisé par descente de gradient stochastique (`SGDClassifier`), configuré pour agir comme un Support Vector Machine (SVM) linéaire optimisé.

## 🚀 Installation

Ce projet utilise [**uv**](https://docs.astral.sh/uv/) (un outil moderne et très rapide de gestion de paquets Python) et requiert **Python 3.11 ou plus récent**.

### Option 1 : Avec `uv` (Recommandé)

1. Clonez le dépôt et naviguez dans le dossier du projet :
   ```bash
   git clone <URL_DU_DEPOT>
   cd rain_prediction
   ```
2. Installez les dépendances et créez l'environnement virtuel de manière transparente :
   ```bash
   uv sync
   ```

### Option 2 : Avec `pip` (Standard)

1. Clonez le dépôt.
2. Créez un environnement virtuel :
   ```bash
   python -m venv .venv
   ```
3. Activez l'environnement virtuel :
   - Sous Windows : `.venv\Scripts\activate`
   - Sous Linux/Mac : `source .venv/bin/activate`
4. Installez les dépendances à partir de `pyproject.toml` :
   ```bash
   pip install -e .
   ```

## 💻 Utilisation

### 1. Entraîner un modèle

Utilisez le script `run.py` en spécifiant le modèle à entraîner via l'argument `--model`. Les identifiants disponibles sont configurés dans `model_registry.py` (ex: `logistic`, `mlp`, `sgd`).

```bash
# Avec uv (assure l'exécution dans le bon environnement)
uv run python run.py --model logistic

# Ou avec l'environnement virtuel classique déjà activé
python run.py --model logistic
```

*Comportement du script* : Il chargera les données spécifiées dans `param.yaml`, effectuera le prétraitement et la validation croisée, lancera la recherche des meilleurs hyperparamètres (si activé dans le fichier yaml), évaluera les performances sur le jeu de test (données non vues), et sauvegardera le modèle sérialisé ainsi que les courbes d'évaluation dans `model_output/`.

### 2. Lancer le Simulateur Interactif (Dashboard)

Une fois qu'au moins un modèle a été entraîné (et sauvegardé dans `model_output/`), vous pouvez lancer l'interface utilisateur pour tester concrètement les prédictions :

```bash
# Avec uv
uv run streamlit run app.py

# Ou avec l'environnement virtuel classique déjà activé
streamlit run app.py
```

L'application s'ouvrira dans votre navigateur web. Vous pourrez y :
- Sélectionner le modèle de votre choix (parmi ceux que vous avez entraînés).
- Consulter rapidement ses métriques de performance globales (Accuracy, ROC-AUC) et son rapport détaillé.
- Ajuster les paramètres météorologiques (pression, humidité, vent, soleil) via des sliders pour voir s'il y a une prédiction de pluie en temps réel, incluant la probabilité d'averse.

### 3. Explorer les données et notebooks

Pour consulter ou modifier les notebooks d'analyse (fichiers `.ipynb` dans le dossier `notebooks/`) :

```bash
uv run jupyter notebook
```

## ⚙️ Configuration Centralisée (`param.yaml`)

Toute la configuration comportementale du projet se trouve dans `param.yaml`. Cela permet de tout contrôler sans avoir à toucher au code source :
- **Variables** : Choix des colonnes cibles, numériques, et catégoriques.
- **Chemins** : Configuration des répertoires de données (`data_dir`) et de sauvegarde des modèles (`model_output_dir`).
- **Optimisation** : Activation ou désactivation globale du GridSearch (`do_grid_search: true/false`).
- **Hyperparamètres** : Définition des grilles de recherche d'hyperparamètres (GridSearch) à explorer pour chaque modèle spécifique.

## 🤝 Contribution & Extension

L'architecture du projet a été pensée pour être facilement extensible à de nouveaux modèles d'intelligence artificielle :

1. Créez un nouveau fichier dans `models/` (ex: `random_forest.py`).
2. Créez une classe qui hérite de `BaseModel` et implémentez les méthodes requises par le contrat de la classe de base (typiquement définir le constructeur et, si besoin, adapter `prepare()`).
3. Ajoutez votre modèle dans le dictionnaire `MODEL_REGISTRY` du fichier `models/model_registry.py`.
4. Et c'est tout ! Le modèle sera automatiquement disponible via le CLI (script `run.py`) et l'interface web Streamlit.
