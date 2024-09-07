# src/intendai/main.py

import os
from src.intendai.pipeline.intent_prediction_pipeline import IntentPredictionPipeline


def main():
     
    # Initialiser le pipeline
    pipeline = IntentPredictionPipeline()

    # Spécifiez le chemin complet vers le fichier JSON
    json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'intents_data.json')

    # Charger les données depuis le fichier JSON
    data, labels = pipeline.load_training_data_from_json(json_path)

    # Ajouter les données d'entraînement
    pipeline.add_training_data(data, labels)

    # Entraîner le modèle
    pipeline.train_model()

    # save model
    pipeline.save_model('./saved_model')


if __name__ == "__main__":
    main()
