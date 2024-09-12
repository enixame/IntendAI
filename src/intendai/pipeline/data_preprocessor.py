# src/intendai/pipeline/data_preprocessor.py

import pandas as pd
from sklearn.utils import resample

class DataPreprocessor:
    """
    Classe pour gérer la préparation et l'encodage des données.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


    def encode_data(self, texts, max_length=128):
        """
        Encode les données textuelles en utilisant le tokenizer.
        
        Args:
            texts (list): Liste de phrases à encoder.
            max_length (int, optionnel): Longueur maximale des séquences. Défaut : 128.
            
        Returns:
            dict: Encodage des textes sous forme de tenseurs.
        """
        return self.tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    
    def decode_data(self, encodings):
        """
        Décode les encodages en phrases originales.
        
        Args:
            encodings (dict): Encodages des phrases générés par le tokenizer.
            
        Returns:
            list: Liste des phrases d'origine.
        """
        return self.tokenizer.batch_decode(encodings['input_ids'], skip_special_tokens=True)


    def balance_data(self, data, labels):
        """
        Fonction pour équilibrer les classes en rééchantillonnant les données.
        """
        df = pd.DataFrame({'text': data, 'label': labels})
        max_samples = df['label'].value_counts().max()

        df_balanced = pd.concat([
            resample(df[df['label'] == label], 
                     replace=True, n_samples=max_samples, random_state=42)
            for label in df['label'].unique()
        ])

        return df_balanced['text'].tolist(), df_balanced['label'].tolist()
