# 🌦️ Rain Prediction Australia

Ce projet est une application complète de Machine Learning visant à prédire s'il pleuvra le lendemain en Australie (`RainTomorrow`). Il intègre un pipeline d'entraînement automatisé, une optimisation des hyperparamètres et un simulateur interactif pour visualiser les prédictions en temps réel.

## 📖 Contexte du Projet

Le climat australien est caractérisé par une forte variabilité. Prédire les précipitations est un défi majeur pour l'agriculture et la gestion des ressources. Ce projet utilise un dataset historique contenant plus de 10 ans de relevés météorologiques quotidiens provenant de nombreuses stations à travers l'Australie.

L'objectif est de fournir un outil robuste capable de :
1. **Traiter et nettoyer** les données météorologiques complexes.
2. **Entraîner plusieurs modèles** d'IA (Régression Logistique, Réseaux de Neurones, SVM) de manière modulaire.
3. **Optimiser** ces modèles via une recherche d'hyperparamètres (GridSearch).
4. **Offrir une interface interactive** permettant à un utilisateur de simuler des conditions météo et d'obtenir une prédiction immédiate.

---

## 📂 Structure du Projet (Dossiers)

Voici le détail de l'organisation du dépôt :

- **`data/`** : Le cœur des données.
    - `raw/` : Contient les fichiers CSV originaux (données brutes).
    - `processed/` : Contient les données nettoyées, transformées et prêtes pour l'entraînement.
- **`models/`** : Contient l'architecture logicielle des modèles.
    - `base_model.py` : La classe mère `BaseModel` qui définit le contrat pour tous les modèles (train, predict, evaluate, etc.).
    - `regression_logistique.py`, `mlp.py`, `sgd.py` : Implémentations spécifiques des algorithmes.
    - `model_registry.py` : Point central pour accéder aux modèles disponibles.
- **`model_output/`** : Résultats des exécutions.
    - Stocke les fichiers `.pkl` (modèles sauvegardés).
    - Contient les rapports d'expérimentation (`report_*.txt`).
    - Contient les courbes de performance (`*_courbes_auc.png`).
- **`src/`** : Logique métier.
    - Contient notamment `training.py` pour orchestrer le processus d'entraînement.
- **`utils/`** : Boîte à outils.
    - `Functions.py` : Fonctions d'aide pour le split des données, le calcul des métriques et la génération des graphiques.
- **`notebooks/`** : Laboratoire d'exploration.
    - Contient les analyses exploratoires (EDA) au format Jupyter Notebook.
- **`app.py`** : Code source de l'application interactive Streamlit.
- **`run.py`** : Script d'interface en ligne de commande (CLI) pour piloter le projet.
- **`param.yaml`** : Fichier de configuration unique pour gérer les hyperparamètres et les features.

---

## 🚀 Guide d'Exécution

### 1. Installation

Assurez-vous d'avoir Python 3.11+ installé. Nous recommandons l'usage de `uv` pour une installation rapide :

```bash
# Avec uv (recommandé)
uv sync

# Ou avec pip standard
python -m venv .venv
source .venv/bin/activate  # (ou .venv\Scripts\activate sur Windows)
pip install -r requirements.txt
```

### 2. Entraîner et Évaluer les Modèles

Pour entraîner un modèle, utilisez le script `run.py` avec l'argument `--model`.

**Modèles disponibles :** `logistic`, `mlp`, `sgd`.

```bash
# Exemple pour entraîner le Multi-Layer Perceptron
python run.py --model mlp
```

**Que fait ce script ?**
1. Il charge les données depuis `data/processed/`.
2. Il prépare les données (scaling, encodage) selon les besoins du modèle.
3. Il lance un **GridSearch** (si `do_grid_search: true` dans `param.yaml`) pour trouver les meilleurs réglages.
4. Il effectue une **validation croisée** rigoureuse.
5. Il évalue le modèle sur un jeu de test indépendant.
6. Il sauvegarde le modèle final et un rapport détaillé dans `model_output/`.

### 3. Lancer l'Application (Simulateur)

Une fois un modèle entraîné, lancez l'interface graphique :

```bash
streamlit run app.py
```

---

## 🎮 Comment utiliser l'Application ?

L'application Streamlit est conçue pour être intuitive. Voici les étapes pour bien l'utiliser :

1. **Configuration (Barre latérale gauche)** :
    - **Choisir le modèle IA** : Sélectionnez le modèle que vous souhaitez tester parmi ceux déjà entraînés.
    - **Performance** : Visualisez instantanément les scores d'accuracy et de ROC-AUC du modèle sélectionné.
2. **📍 Conditions Actuelles (Centre)** :
    - Utilisez les **sliders (curseurs)** pour modifier les paramètres météo :
        - *Température* : Ajustez les températures du matin et de l'après-midi.
        - *Humidité & Pression* : Modifiez les taux d'humidité et la pression atmosphérique (facteurs clés).
        - *Ensoleillement & Vent* : Ajustez la vitesse du vent et le nombre d'heures de soleil.
3. **🔮 Résultat de la Prédiction** :
    - Le résultat s'affiche en temps réel.
    - **Probabilité de pluie** : Une jauge vous indique la confiance du modèle.
    - **Recommandation** : Un message clair s'affiche ("Prévoyez de la pluie !" ou "Journée ensoleillée en vue").
4. **📉 Analyse des Performances** :
    - En bas de page, consultez les courbes ROC et Precision-Recall pour comprendre la fiabilité historique du modèle.

---

## ⚙️ Personnalisation

Pour modifier les paramètres de recherche (GridSearch) ou changer les colonnes utilisées par l'IA, éditez simplement le fichier `param.yaml`. Aucun changement de code n'est nécessaire pour ajuster les hyperparamètres !
