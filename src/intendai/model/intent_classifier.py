# src/intendai/model/intent_classifier.py

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class IntentClassifier:
    """Responsable de la création, de l'entraînement et de la prédiction des intentions."""

    def __init__(self):
        self.model = MultinomialNB()

    def train(self, X_train, y_train):
        """Entraîne le modèle de classification."""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Prédit les intentions des nouvelles phrases."""
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Évalue le modèle de classification."""
        y_pred = self.model.predict(X_test)
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
