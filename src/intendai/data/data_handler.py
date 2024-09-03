# src/data/data_handler.py

import pandas as pd

class DataHandler:
    """Responsable de la gestion des données."""

    def __init__(self):
        self.df = pd.DataFrame(columns=['Phrase', 'Intention'])

    def add_data(self, phrases, intentions):
        """Ajoute de nouvelles phrases et intentions au DataFrame."""
        new_data = pd.DataFrame({'Phrase': phrases, 'Intention': intentions})
        self.df = pd.concat([self.df, new_data], ignore_index=True)

    def get_data(self):
        """Retourne le DataFrame actuel des données."""
        return self.df
