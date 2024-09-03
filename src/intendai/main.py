# src/intendai/main.py

from src.intendai.pipeline.intent_prediction_pipeline import IntentPredictionPipeline

def main():
    # Initialiser le pipeline
    pipeline = IntentPredictionPipeline()

    # Ajout de données d'entraînement pour les salutations
    greetings_data = [
        "Salut",
        "Bonjour",
        "Hey",
        "Bonsoir",
        "Coucou",
        "Bonjour à tous",
        "Hey, salut",
        "Yo",
        "Hello",
        "Salut tout le monde"
    ]

    greetings_intentions = ['greetings'] * len(greetings_data)

    # Ajout de données d'entraînement pour l'état de santé
    health_status_data = [
        "Comment ça va ?",
        "Ça va bien ?",
        "Comment te sens-tu ?",
        "La forme ?",
        "Ça va, toi ?",
        "Tu te sens bien ?",
        "Tout va bien ?",
        "Comment va ta journée ?",
        "Ça roule ?",
        "Comment tu vas aujourd'hui ?"
    ]

    health_status_intentions = ['health_status'] * len(health_status_data)

    # Ajouter les deux ensembles de données au pipeline
    pipeline.add_training_data(greetings_data, greetings_intentions)
    pipeline.add_training_data(health_status_data, health_status_intentions)

    # Entraîner le modèle
    pipeline.train_model()

    # Prédire une intention
    print(pipeline.predict_intent("Salut"))  # Devrait retourner "greetings"
    print(pipeline.predict_intent("Comment ça va ?"))  # Devrait retourner "health_status"
    print(pipeline.predict_intent("Bonsoir"))  # Devrait retourner "greetings"
    print(pipeline.predict_intent("Ça va bien ?"))  # Devrait retourner "health_status"
    print(pipeline.predict_intent("Hey ma gueule ! bien ou bien ?"))  # Devrait retourner "health_status"

if __name__ == "__main__":
    main()
