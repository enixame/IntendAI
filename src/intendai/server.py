# src/intendai/server.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from src.intendai.pipeline.intent_prediction_pipeline import IntentPredictionPipeline

app = Flask(__name__)
CORS(app)  # Permet de gérer les requêtes cross-origin entre React et Flask

# Charger le modèle sauvegardé
pipeline = IntentPredictionPipeline()
pipeline.load_model("./saved_model")


@app.route('/predict', methods=['GET'])
def get_intention():
    """
    Récupère l'intention d'une phrase à partir d'une requête GET.
    Paramètres :
    - phrase : La phrase pour laquelle on veut obtenir l'intention.
    - threshold : Seuil de confiance pour l'intention (par défaut 0.7).
    """
    phrase = request.args.get('phrase')
    threshold = float(request.args.get('threshold', 0.7))

    if not phrase:
        return jsonify({"error": "A phrase must be provided."}), 400

    # Utiliser le pipeline pour prédire l'intention
    prediction = pipeline.predict_intent(phrase, threshold=threshold)

    return jsonify({"intention": prediction})


@app.route('/train', methods=['POST'])
def train_model():
    """
    Entraîne le modèle de manière incrémentale à partir d'un fichier JSON fourni en POST.
    JSON Format attendu :
    {
        "intents": {
            "label1": ["phrase1", "phrase2"],
            "label2": ["phrase3", "phrase4"]
        }
    }
    """
    # Charger les données JSON depuis la requête POST
    json_data = request.get_json()

    if not json_data:
        return jsonify({"error": "No data provided"}), 400

    phrases, labels = [], []
    for intent, phrase_list in json_data.get("intents", {}).items():
        phrases.extend(phrase_list)
        labels.extend([intent] * len(phrase_list))

    if not phrases or not labels:
        return jsonify({"error": "Invalid data format"}), 400

    # Encoder les labels avec le LabelEncoder utilisé dans le pipeline
    labels_encoded = pipeline.label_encoder.transform(labels)

    # Entraînement incrémental avec les nouvelles données
    pipeline.train_model(incremental=True, new_data=phrases, new_labels=labels_encoded)

    return jsonify({"message": "Training completed successfully"})


@app.route('/save_model', methods=['POST'])
def save_model():
    """
    Sauvegarde le modèle entraîné dans le répertoire spécifié.
    Le répertoire est envoyé dans le corps de la requête POST.
    """
    data = request.get_json()

    # Vérifier que le répertoire est fourni
    if not data or 'model_dir' not in data:
        return jsonify({"error": "A model directory must be provided."}), 400

    model_dir = data['model_dir']

    # Sauvegarder le modèle
    try:
        pipeline.save_model(model_dir)
        return jsonify({"message": f"Model saved successfully in '{model_dir}'"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Lancer le serveur
    app.run(host='0.0.0.0', port=5000, debug=True)
