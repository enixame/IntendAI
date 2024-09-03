# IntendAI
IntendAI est un projet de machine learning conçu pour classer les intentions à partir de phrases en langage naturel.

## Structure du Projet

- **src/intendai/data/data_handler.py**: Contient la classe `DataHandler` qui gère la collecte et la gestion des données.
- **src/intendai/preprocessing/preprocessor.py**: Contient la classe `Preprocessor` pour le prétraitement du texte.
- **src/intendai/model/intent_classifier.py**: Contient la classe `IntentClassifier` pour l'entraînement et la prédiction des intentions.
- **src/intendai/pipeline/intent_prediction_pipeline.py**: Contient la classe `IntentPredictionPipeline` qui orchestre l'ensemble du processus de classification des intentions.
- **src/intendai/main.py**: Script principal pour exécuter le pipeline.
- **README.md**: Documentation du projet.
- **requirements.txt**: Liste des dépendances nécessaires pour exécuter le projet.

## Installation

1. Clonez ce dépôt.
2. Installez les dépendances nécessaires avec pip:

```bash
pip install -r requirements.txt
