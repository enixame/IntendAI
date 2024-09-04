# src/intendai/main.py

from src.intendai.pipeline.intent_prediction_pipeline import IntentPredictionPipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import pandas as pd

# Toutes les données d'intention
greetings_data = [
    "Salut", "Bonjour", "Hey", "Bonsoir", "Coucou", "Bonjour à tous", "Hey, salut", "Yo", "Hello", "Salut tout le monde"
    # Ajoute plus d'exemples ici...
]
greetings_intentions = ['greetings'] * len(greetings_data)

health_status_data = [
    "Comment ça va ?", "Ça va bien ?", "Comment te sens-tu ?", "La forme ?", "Ça va, toi ?", "Tu te sens bien ?", "Tout va bien ?"
    # Ajoute plus d'exemples ici...
]
health_status_intentions = ['health_status'] * len(health_status_data)

status_data = [
    "Comment se passe ton stream Twitch ?", "Quel est le statut de ta vidéo ?", "As-tu des nouvelles sur ta chaîne Twitch ?", "Comment avance ton projet ?"
    # Ajoute plus d'exemples ici...
]
status_intentions = ['status'] * len(status_data)

backseat_data = [
    "Tu devrais vraiment prendre à gauche ici.", "Pourquoi ne fais-tu pas ça à ma manière ?", "Tu devrais essayer de le faire comme ça."
    # Ajoute plus d'exemples ici...
]
backseat_intentions = ['backseat'] * len(backseat_data)

bad_data = [
    "C'est vraiment nul ce que tu fais.", "Tu es incompétent.", "C'est la pire chose que j'ai jamais vue."
    # Ajoute plus d'exemples ici...
]
bad_intentions = ['bad'] * len(bad_data)

unknown_data = [
    "asdfghjkl", "1234567890", "!@#$%^&*()", "qwertyuiop", "zxcvbnm,./"
    # Ajoute plus d'exemples ici...
]
unknown_intentions = ['unknown'] * len(unknown_data)

data = greetings_data + health_status_data + status_data + backseat_data + bad_data + unknown_data
labels = greetings_intentions + health_status_intentions + status_intentions + backseat_intentions + bad_intentions + unknown_intentions

def balance_data(data, labels):
    df = pd.DataFrame({'text': data, 'label': labels})
    max_samples = df['label'].value_counts().max()

    df_balanced = pd.concat([
        resample(df[df['label'] == label], 
                 replace=True, 
                 n_samples=max_samples, 
                 random_state=42)
        for label in df['label'].unique()
    ])
    
    return df_balanced['text'].tolist(), df_balanced['label'].tolist()

def prepare_data(data, labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder

def main():
    # Rééchantillonner les données pour équilibrer les classes
    data_balanced, labels_balanced = balance_data(data, labels)

    # Initialiser le pipeline
    pipeline = IntentPredictionPipeline()

    # Préparer les données d'entraînement et de test avec des données équilibrées
    X_train, X_test, y_train, y_test, label_encoder = prepare_data(data_balanced, labels_balanced)
    pipeline.add_training_data(X_train, y_train)

    # Entraîner le modèle
    pipeline.train_model()

    # Évaluer la précision sur l'ensemble de test
    predictions = [pipeline.predict_intent(text) for text in X_test]
    accuracy = sum(pred == label_encoder.inverse_transform([true])[0] for pred, true in zip(predictions, y_test)) / len(y_test)
    print(f'Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    main()
