import os
import pytest
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from src.intendai.pipeline.intent_prediction_pipeline import IntentPredictionPipeline


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
