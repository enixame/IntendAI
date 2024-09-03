# src/intendai/preprocessing/preprocessor.py

from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessor:
    """Responsable du prétraitement des données textuelles."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, X):
        """Entraîne le vectoriseur TF-IDF et transforme les données d'entraînement."""
        return self.vectorizer.fit_transform(X)

    def transform(self, X):
        """Transforme les nouvelles données en utilisant le vectoriseur entraîné."""
        return self.vectorizer.transform(X)
