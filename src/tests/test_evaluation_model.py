import os
import pytest
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from src.intendai.pipeline.intent_prediction_pipeline import IntentPredictionPipeline


# Fonction d'évaluation du modèle
def evaluate_model(pipeline, data, labels, label_encoder):
    """
    Évalue le modèle sur un ensemble de données de test.

    Args:
        pipeline (IntentPredictionPipeline): Pipeline de prédiction d'intention.
        data (list): Liste des phrases pour le test.
        labels (list): Liste des labels encodés pour le test.
        label_encoder (LabelEncoder): L'encodeur des labels.

    Prints:
        Matrice de confusion et rapport de classification.
    """

    # Ajouter 'unknown' à l'encodeur de labels s'il ne l'a pas déjà
    if 'unknown' not in label_encoder.classes_:
        # Étendre les classes du label_encoder avec "unknown"
        classes_with_unknown = list(label_encoder.classes_) + ['unknown']
        label_encoder.classes_ = np.array(classes_with_unknown)

    # Prédire les intentions pour les données de test
    predictions = [pipeline.predict_intent(text) for text in data]

    # Décoder les labels et les prédictions pour avoir les noms de classes au lieu de 0 et 1
    labels_decoded = label_encoder.inverse_transform(labels)
    predictions_decoded = np.array(predictions)

    # Obtenir les classes présentes dans les labels et les prédictions
    unique_labels = np.unique(labels_decoded)
    unique_predictions = np.unique(predictions_decoded)

    # Obtenir toutes les classes présentes dans le jeu de test
    all_classes = np.union1d(unique_labels, unique_predictions)

    # Calcul de la matrice de confusion
    cm = confusion_matrix(labels_decoded, predictions_decoded, labels=all_classes)
    print("Matrice de confusion :")
    print(cm)

    # Rapport de classification avec les classes présentes dans le jeu de test
    print("\nRapport de classification :")
    print(classification_report(labels_decoded, predictions_decoded, target_names=all_classes))



# Fixture pour initialiser le pipeline
@pytest.fixture(scope="module")
def pipeline():
    # Initialiser le pipeline
    pipeline = IntentPredictionPipeline()
    pipeline.load_model('./saved_model')
    
    return pipeline



# Test pour évaluer le modèle
def test_evaluate_model(pipeline):
    """
    Teste l'évaluation du modèle avec des données de test.

    Args:
        pipeline (IntentPredictionPipeline): Pipeline de prédiction d'intention.
    """
    # Spécifiez le chemin complet vers le fichier JSON
    json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'intents_data.json')

    # Charger les données depuis le fichier JSON
    data, labels_encoded = pipeline.load_training_data_from_json(json_path)

    # Utiliser une partie des données pour l'évaluation (par exemple, 20% pour les tests)
    split_index = int(0.8 * len(data))
    X_test, y_test = data[split_index:], labels_encoded[split_index:]

    # Évaluer le modèle
    evaluate_model(pipeline, X_test, y_test, pipeline.label_encoder)
