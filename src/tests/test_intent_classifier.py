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

    # Créer un label encoder temporaire avec la classe "unknown"
    extended_label_encoder = LabelEncoder()
    extended_label_encoder.fit(classes_with_unknown)

    predictions = [pipeline.predict_intent(text) for text in data]
    predictions_encoded = extended_label_encoder.transform(predictions)

    # Décoder les labels et les prédictions pour avoir les noms de classes au lieu de 0 et 1
    labels_decoded = label_encoder.inverse_transform(labels)
    predictions_decoded = extended_label_encoder.inverse_transform(predictions_encoded)

    # Calcul de la matrice de confusion
    cm = confusion_matrix(labels_decoded, predictions_decoded, labels=extended_label_encoder.classes_)
    print("Matrice de confusion :")
    print(cm)

    # Rapport de classification avec toutes les classes présentes
    print("\nRapport de classification :")
    print(classification_report(labels_decoded, predictions_decoded, target_names=extended_label_encoder.classes_, labels=np.unique(labels)))


# Fixture pour initialiser le pipeline
@pytest.fixture(scope="module")
def pipeline():
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
    data, labels = pipeline.load_training_data_from_json(json_path)

    # Utiliser le label encoder du pipeline (celui utilisé à l'entraînement)
    labels_encoded = pipeline.label_encoder.fit_transform(labels)

    # Utiliser une partie des données pour l'évaluation (par exemple, 20% pour les tests)
    split_index = int(0.8 * len(data))
    X_test, y_test = data[split_index:], labels_encoded[split_index:]

    # Évaluer le modèle
    evaluate_model(pipeline, X_test, y_test, pipeline.label_encoder)


# Test des prédictions individuelles
def test_phrases(pipeline):
    """
    Teste quelques phrases spécifiques pour voir les prédictions du modèle.

    Args:
        pipeline (IntentPredictionPipeline): Pipeline de prédiction d'intention.
    """
    test_phrases = [
        "Bonjour", 
        "Salut ma gueule !", 
        "Wesh ma guele !",
        "Comment ça va ?", 
        "Tu devrais ralentir ici.", 
        "Vas à gauche.",
        "C'est vraiment nul ce que tu fais.", 
        "T'es nul",
        "Aujourd'hui, il fait chaud. je suis sorti prendre un verre.",
        "Je travaille dans le BTP."
        "c'est ça",
        "tout à fait",
        "c'est cool",
        "on est sur quel jeu ?",
        "c'est quoi le jeu ?",
        "Peux-tu m'envoyer le document, s'il te plaît ?",
        "!!eeefffsqqsezeaazeeazae))))))((''''((((('''(()))))))))"
    ]
    
    for phrase in test_phrases:
        print(f"Phrase: '{phrase}' - Prédiction: '{pipeline.predict_intent(phrase)}'")

# Tests spécifiques à chaque intention
def test_greetings(pipeline):
    """
    Teste la prédiction pour l'intention "greetings".

    Args:
        pipeline (IntentPredictionPipeline): Pipeline de prédiction d'intention.
    """
    phrases = ["Bonjour", "Salut", "Hey", "hello", "salut à tous", "salut tout le monde"]
    for phrase in phrases:
        assert pipeline.predict_intent(phrase) == "greetings"

def test_health_status(pipeline):
    """
    Teste la prédiction pour l'intention "health_status".

    Args:
        pipeline (IntentPredictionPipeline): Pipeline de prédiction d'intention.
    """
    phrases = ["Comment ça va ?", "Ça va bien ?", "La forme ?"]
    for phrase in phrases:
        assert pipeline.predict_intent(phrase) == "health_status"

def test_backseat(pipeline):
    """
    Teste la prédiction pour l'intention "backseat".

    Args:
        pipeline (IntentPredictionPipeline): Pipeline de prédiction d'intention.
    """
    phrases = ["Va à gauche, c'est plus rapide.", "Utilise la potion maintenant."]
    for phrase in phrases:
        assert pipeline.predict_intent(phrase) == "backseat"

def test_bad(pipeline):
    """
    Teste la prédiction pour l'intention "bad".

    Args:
        pipeline (IntentPredictionPipeline): Pipeline de prédiction d'intention.
    """
    phrases = ["C'était nul !", "Tu es incompétent."]
    for phrase in phrases:
        assert pipeline.predict_intent(phrase) == "bad"
