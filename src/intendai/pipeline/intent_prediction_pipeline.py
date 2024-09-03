# src/intendai/pipeline/intent_prediction_pipeline.py

from src.intendai.data.data_handler import DataHandler
from src.intendai.preprocessing.preprocessor import Preprocessor
from src.intendai.model.intent_classifier import IntentClassifier
from sklearn.model_selection import train_test_split

class IntentPredictionPipeline:
    """Responsable de l'orchestration des étapes du pipeline pour prédire les intentions."""

    def __init__(self):
        self.data_handler = DataHandler()
        self.preprocessor = Preprocessor()
        self.classifier = IntentClassifier()

    def add_training_data(self, phrases, intentions):
        """Ajoute des données d'entraînement au pipeline."""
        self.data_handler.add_data(phrases, intentions)

    def train_model(self):
        """Entraîne le modèle avec les données disponibles."""
        df = self.data_handler.get_data()
        X = df['Phrase']
        y = df['Intention']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_vect = self.preprocessor.fit_transform(X_train)
        X_test_vect = self.preprocessor.transform(X_test)
        self.classifier.train(X_train_vect, y_train)
        self.classifier.evaluate(X_test_vect, y_test)

    def predict_intent(self, phrase):
        """Prédit l'intention d'une phrase donnée."""
        X_vect = self.preprocessor.transform([phrase])
        return self.classifier.predict(X_vect)[0]
