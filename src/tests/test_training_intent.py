import os
import json
import pytest
from src.intendai.pipeline.intent_prediction_pipeline import IntentPredictionPipeline

def load_intent_data(json_path):
    """
    Charge les données d'intentions à partir d'un fichier JSON.

    Args:
        json_path (str): Chemin vers le fichier JSON contenant les intentions.

    Returns:
        Tuple (list, list): Liste des phrases et des labels encodés.
    """
    try:
        # Charger les données à partir du fichier JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        phrases = []
        labels = []

        # Parcourir les intentions et extraire les phrases et labels
        for intent, phrases_list in data["intents"].items():
            if phrases_list:  # Vérifier si la liste n'est pas vide
                phrases.extend(phrases_list)
                labels.extend([intent] * len(phrases_list))

        # Vérifier si des données sont présentes
        if len(phrases) == 0:
            raise ValueError("Le fichier JSON ne contient aucune donnée de formation.")

        return phrases, labels

    except FileNotFoundError:
        print(f"Erreur : le fichier '{json_path}' n'existe pas.")
    except json.JSONDecodeError:
        print(f"Erreur : le fichier '{json_path}' n'est pas un fichier JSON valide.")
    except Exception as e:
        print(f"Une erreur inattendue s'est produite : {str(e)}")


# Fixture pour initialiser le pipeline
@pytest.fixture(scope="module")
def pipeline():
    # Initialiser le pipeline
    pipeline = IntentPredictionPipeline()
    pipeline.load_model('./saved_model')
    
    return pipeline


# Tests spécifiques à chaque intention
def test_health_status_with_training(pipeline):
    """
    Teste la prédiction pour l'intention "health_status".

    Args:
        pipeline (IntentPredictionPipeline): Pipeline de prédiction d'intention.
    """
    phrases = ["la forme ?"]
    for phrase in phrases:
        prediction = pipeline.predict_intent(phrase)
        print(f"Phrase: '{phrase}' - Prédiction: '{prediction}'")

    json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'new_health_status.json')
    new_phrases, new_labels = load_intent_data(json_path)

    if new_phrases and new_labels:
        # Encodage des labels avec le LabelEncoder utilisé dans le pipeline
        new_labels_encoded = pipeline.label_encoder.transform(new_labels)

        # Entraînement incrémental avec les nouvelles données
        pipeline.train_model(incremental=True, new_data=new_phrases, new_labels=new_labels_encoded)

    for phrase in phrases:
        prediction = pipeline.predict_intent(phrase)
        print(f"Phrase: '{phrase}' - Prédiction: '{prediction}'")
    assert prediction == "health_status"



def test_backseat_with_training(pipeline):
    """
    Teste la prédiction pour l'intention "backseat".

    Args:
        pipeline (IntentPredictionPipeline): Pipeline de prédiction d'intention.
    """
    phrases = ["Prends à gauche, t'as manqué quelque chose !", "Pourquoi tu refuses de prendre cette arme ? Elle est beaucoup mieux.", "Non, essaie ça à la place, c'est bien plus efficace.",
               "Tu fais n'importe quoi, c'est pas du tout la bonne façon de jouer.", "Si t'avais suivi mes conseils, t'aurais déjà gagné.",
               "T'es vraiment sûr que c'est la bonne direction à prendre ?", "Sérieusement, tu rates plein de trucs, explore un peu mieux !"]
    
    for phrase in phrases:
        prediction = pipeline.predict_intent(phrase)
        print(f"Phrase: '{phrase}' - Prédiction: '{prediction}'")

    json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'new_backseats.json')
    new_phrases, new_labels = load_intent_data(json_path)

    if new_phrases and new_labels:
        # Encodage des labels avec le LabelEncoder utilisé dans le pipeline
        new_labels_encoded = pipeline.label_encoder.transform(new_labels)

        # Entraînement incrémental avec les nouvelles données
        pipeline.train_model(incremental=True, new_data=new_phrases, new_labels=new_labels_encoded)

    for phrase in phrases:
        prediction = pipeline.predict_intent(phrase)
        print(f"Phrase: '{phrase}' - Prédiction: '{prediction}'")
        assert prediction == "backseat"