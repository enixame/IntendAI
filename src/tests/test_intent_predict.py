import pytest
from src.intendai.pipeline.intent_prediction_pipeline import IntentPredictionPipeline


# Fixture pour initialiser le pipeline
@pytest.fixture(scope="module")
def pipeline():
    # Initialiser le pipeline
    pipeline = IntentPredictionPipeline()
    pipeline.load_model('./saved_model')
    
    return pipeline


# Tests spécifiques à chaque intention
def test_greetings(pipeline):
    """
    Teste la prédiction pour l'intention "greetings".

    Args:
        pipeline (IntentPredictionPipeline): Pipeline de prédiction d'intention.
    """
    phrases = ["Bonjour", "Salut", "Hey", "hello", "salut à tous", "salut tout le monde !", "coucou", "wesh", "salut le bot, tu vas bien ?"]
    for phrase in phrases:
        prediction = pipeline.predict_intent(phrase)
        print(f"Phrase: '{phrase}' - Prédiction: '{prediction}'")
        assert prediction == "greetings"

    
def test_health_status(pipeline):
    """
    Teste la prédiction pour l'intention "health_status".

    Args:
        pipeline (IntentPredictionPipeline): Pipeline de prédiction d'intention.
    """
    phrases = ["Comment ça va ?", "Ça va bien ?", "est-ce que tout va bien ?", "tu vas bien ?"]
    for phrase in phrases:
        prediction = pipeline.predict_intent(phrase)
        print(f"Phrase: '{phrase}' - Prédiction: '{prediction}'")
        assert prediction == "health_status"



def test_backseat(pipeline):
    """
    Teste la prédiction pour l'intention "backseat".

    Args:
        pipeline (IntentPredictionPipeline): Pipeline de prédiction d'intention.
    """
    phrases = ["Va à gauche, c'est plus rapide.", "Utilise la potion maintenant.", "Prends la compétence."]
    for phrase in phrases:
        prediction = pipeline.predict_intent(phrase)
        print(f"Phrase: '{phrase}' - Prédiction: '{prediction}'")
        assert prediction == "backseat"

    
def test_bad(pipeline):
    """
    Teste la prédiction pour l'intention "bad".

    Args:
        pipeline (IntentPredictionPipeline): Pipeline de prédiction d'intention.
    """
    phrases = ["C'était nul !", "Tu es incompétent.", "tu ne comprends rien", "tu ne sais pas jouer", "t'es nul"]
    for phrase in phrases:
        prediction = pipeline.predict_intent(phrase)
        print(f"Phrase: '{phrase}' - Prédiction: '{prediction}'")
        assert prediction == "bad"


def test_ask(pipeline):
    """
    Teste la prédiction pour l'intention "ask".

    Args:
        pipeline (IntentPredictionPipeline): Pipeline de prédiction d'intention.
    """
    phrases = ["C'est quoi comme jeu ?", "Peux-tu m'aider à faire ?", "Peux-tu m'expliquer comment tu fait ?"]
    for phrase in phrases:
        prediction = pipeline.predict_intent(phrase)
        print(f"Phrase: '{phrase}' - Prédiction: '{prediction}'")
        assert prediction == "ask"


def test_unknown(pipeline):
    """
    Teste la prédiction pour l'intention "unknown".

    Args:
        pipeline (IntentPredictionPipeline): Pipeline de prédiction d'intention.
    """
    phrases = ["))))))))('('('('('('()))))))"]
    for phrase in phrases:
        prediction = pipeline.predict_intent(phrase)
        print(f"Phrase: '{phrase}' - Prédiction: '{prediction}'")
        assert prediction == "unknown"