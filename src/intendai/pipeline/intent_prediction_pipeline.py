# src/intendai/pipeline/intent_prediction_pipeline.py

import json
import numpy as np
import os
import pandas as pd
import pickle
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import DebertaTokenizer, DebertaForSequenceClassification, TrainingArguments
from transformers import Trainer

from .custom_dataset import CustomDataset
from .data_preprocessor import DataPreprocessor
from .loss_strategy import WeightedCrossEntropyLoss
from .metrics_strategy import WeightedMetrics


class CustomTrainer(Trainer):
    """
    Custom Trainer pour gérer le comportement de drop_last dans les DataLoader et utiliser une stratégie de perte personnalisée.
    """
    
    def __init__(self, *args, loss_strategy=None, **kwargs):
        """
        Initialisation du CustomTrainer.

        Args:
            loss_strategy (LossStrategy, optionnel): Stratégie personnalisée pour le calcul de la perte.
        """
        super().__init__(*args, **kwargs)
        self.loss_strategy = loss_strategy  # Stocker la stratégie de perte

    def get_train_dataloader(self) -> DataLoader:
        """
        Retourne le DataLoader pour l'entraînement, avec drop_last=True pour éviter les batchs incomplets.
        """
        train_dataset = self.train_dataset
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,  # Mélanger les données d'entraînement
            drop_last=True,  # Laisser tomber le dernier batch s'il est incomplet
            num_workers=self.args.dataloader_num_workers
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """
        Retourne le DataLoader pour l'évaluation, avec drop_last=True.
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            drop_last=True,  # Laisser tomber le dernier batch s'il est incomplet
            num_workers=self.args.dataloader_num_workers
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Fonction pour calculer la perte en utilisant la stratégie de perte personnalisée.
        Si aucune stratégie de perte n'est spécifiée, utilise la perte par défaut.
        """
        # Récupérer les labels des inputs
        labels = inputs.get("labels")

        # Appliquer la stratégie de perte personnalisée si elle est définie
        if self.loss_strategy is not None:
            # Calculer la perte personnalisée via la stratégie définie
            loss = self.loss_strategy.compute_loss(model, inputs)
            if return_outputs:
                return loss, model(**inputs)
            return loss
        else:
            # Utiliser la méthode par défaut si aucune stratégie de perte n'est spécifiée
            return super().compute_loss(model, inputs, return_outputs)


class IntentPredictionPipeline:
    """
    Pipeline de prédiction d'intention utilisant DeBERTa avec stratégies pour la perte et les métriques.
    Cette classe permet de charger les données, entraîner un modèle, et faire des prédictions sur les intentions.
    """

    def __init__(self, loss_strategy=None, metrics_strategy=None, tokenizer=None, model=None):
        """
        Initialise le pipeline avec des stratégies par défaut pour la perte et les métriques.

        Args:
            loss_strategy (LossStrategy, optionnel): Stratégie personnalisée pour le calcul de la perte.
            metrics_strategy (MetricsStrategy, optionnel): Stratégie personnalisée pour le calcul des métriques.
            tokenizer (DebertaTokenizer, optionnel): Tokenizer utilisé pour encodage.
            model (DebertaForSequenceClassification, optionnel): Modèle de classification.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Utiliser des stratégies par défaut si aucune n'est fournie
        self.tokenizer = tokenizer or DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        self.model = model or DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels=7)
        self.model.to(self.device)
        self.model_dir = None

        # Par défaut, utiliser les stratégies WeightedCrossEntropyLoss et WeightedMetrics
        self.loss_strategy = loss_strategy or WeightedCrossEntropyLoss(None)
        self.metrics_strategy = metrics_strategy or WeightedMetrics()

        self.label_encoder = None
        self.class_weights = None
        self.data_preprocessor = DataPreprocessor(self.tokenizer)

        self.unencoded_labels = None  # les labels non encodés chargés depuis un json


    def load_training_data_from_json(self, json_path):
        """
        Charge les données d'entraînement à partir d'un fichier JSON.

        Args:
            json_path (str): Chemin vers le fichier JSON contenant les phrases et intentions.

        Returns:
            Tuple (list, list): Une liste de phrases et une liste de labels encodés.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            phrases = []
            labels = []

            # Extraction des phrases et des labels
            for intent, phrases_list in data["intents"].items():
                phrases.extend(phrases_list)
                labels.extend([intent] * len(phrases_list))

            # Sauvegarde des labels non encodés
            self.unencoded_labels = labels.copy()

            # Encoder les labels avec LabelEncoder
            self.label_encoder = LabelEncoder()
            labels_encoded = self.label_encoder.fit_transform(labels)

            return phrases, labels_encoded

        except FileNotFoundError:
            print(f"Erreur : le fichier '{json_path}' n'existe pas.")
        except json.JSONDecodeError:
            print(f"Erreur : le fichier '{json_path}' n'est pas un fichier JSON valide.")
        except Exception as e:
            print(f"Une erreur inattendue s'est produite : {str(e)}")


    def initialize_data_encodings(self, train_texts, train_labels, val_texts=None, val_labels=None):
        """
        Initialise les encodages des données d'entraînement et de validation.

        Args:
            train_texts (list): Les phrases d'entraînement.
            train_labels (list): Les labels d'entraînement.
            val_texts (list, optionnel): Les phrases de validation.
            val_labels (list, optionnel): Les labels de validation.
        """
        self.train_encodings = self.data_preprocessor.encode_data(train_texts)
        self.train_labels = train_labels
        if val_texts and val_labels:
            self.val_encodings = self.data_preprocessor.encode_data(val_texts)
            self.val_labels = val_labels


    def add_training_data(self, data, labels):
        """
        Prépare les données d'entraînement et de validation pour l'entraînement.

        Args:
            data (list): Liste des phrases.
            labels (list): Liste des labels encodés.
        """
        data_balanced, labels_balanced = self.data_preprocessor.balance_data(data, labels)

        # Diviser en jeux de données d'entraînement et de validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            data_balanced, labels_balanced, test_size=0.2, random_state=42
        )

        # Initialiser les encodages
        self.initialize_data_encodings(train_texts, train_labels, val_texts, val_labels)

        # Calculer les poids de classe
        unique_classes = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels)

        weights = np.ones(self.model.num_labels, dtype=np.float32)
        for i, class_idx in enumerate(unique_classes):
            weights[class_idx] = class_weights[i]

        # Mise à jour des poids de classes
        self.class_weights = torch.tensor(weights, dtype=torch.float).to(self.device)
        self.loss_strategy = WeightedCrossEntropyLoss(self.class_weights)


    def save_old_data(self, data, labels):
        """
        Sauvegarde les anciennes données pour éviter l'oubli catastrophique lors de l'entraînement incrémental.

        Args:
            data (list): Liste des phrases d'entraînement.
            labels (list): Liste des labels correspondants.
        """
        try:
            with open(os.path.join(self.model_dir, 'old_data.pkl'), 'wb') as f:
                pickle.dump((data, labels), f)
            print("Les anciennes données ont été sauvegardées avec succès.")
    
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des anciennes données : {str(e)}")


    def sample_old_data(self, sample_size_per_label=100):
        """
        Récupère un échantillon de phrases équilibré avec 100 exemples par label des anciennes données d'entraînement.

        Args:
            sample_size_per_label (int): Nombre d'exemples à récupérer par label.

        Returns:
            Tuple (list, list): Liste des phrases et des labels d'un échantillon des anciennes données.
        """
        try:
            # Charger les anciennes données si elles sont disponibles
            old_data_path = os.path.join(self.model_dir, 'old_data.pkl')
            if not os.path.exists(old_data_path):
                print("Aucune ancienne donnée trouvée. L'entraînement sera effectué uniquement avec les nouvelles données.")
                return [], []

            with open(old_data_path, 'rb') as f:
                old_data, old_labels = pickle.load(f)

            # Vérification que les données ne sont pas vides
            if not old_data or not old_labels:
                print("Les anciennes données sont vides. L'entraînement sera effectué uniquement avec les nouvelles données.")
                return [], []

            # S'assurer que le nombre de données et de labels est cohérent
            assert len(old_data) == len(old_labels), "Les anciennes données et labels ont des tailles différentes."

            # Créer un DataFrame pour faciliter l'échantillonnage par label
            df = pd.DataFrame({'text': old_data, 'label': old_labels})

            # Échantillonnage fixe de 100 exemples par label
            sampled_df = pd.concat([
                df[df['label'] == label].sample(n=min(len(df[df['label'] == label]), sample_size_per_label), random_state=42)
                for label in df['label'].unique()
            ])

            # Extraire les phrases et labels échantillonnés
            sampled_old_data = sampled_df['text'].tolist()
            sampled_old_labels = sampled_df['label'].tolist()

            return sampled_old_data, sampled_old_labels

        except Exception as e:
            print(f"Erreur lors de l'échantillonnage des anciennes données : {e}")
            return [], []



    def initialize_model_dir(self, default_dir='./saved_model'):
        """
        Initialise self.model_dir si elle n'est pas définie et crée le répertoire si nécessaire.

        Args:
            default_dir (str): Répertoire par défaut pour sauvegarder le modèle et les données.
        """
        if not hasattr(self, 'model_dir') or self.model_dir is None:
            # Si self.model_dir n'est pas défini, utiliser un répertoire par défaut
            self.model_dir = default_dir
            print(f"self.model_dir n'était pas défini. Utilisation de {self.model_dir} comme répertoire de sauvegarde.")

        # Créer le répertoire s'il n'existe pas
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print(f"Répertoire {self.model_dir} créé pour sauvegarder le modèle.")


    def train_model(self, default_dir='./saved_model', incremental=False, new_data=None, new_labels=None):
        """
        Entraîne le modèle sur les données fournies.

        Args:
            default_dir (str): Répertoire par défaut pour sauvegarder le modèle et les données.
            incremental (bool, optionnel): Si vrai, effectue un entraînement incrémental sur de nouvelles données.
            new_data (list, optionnel): Nouvelles phrases pour l'entraînement incrémental.
            new_labels (list, optionnel): Nouveaux labels pour l'entraînement incrémental.
        """
        incremental_training = incremental and new_data is not None and new_labels is not None
        num_train_epochs = 20 if not incremental_training else 5
        batch_size = 16 if not incremental_training else 1
        
        # Initialiser self.model_dir si ce n'est pas fait
        self.initialize_model_dir(default_dir)

        # Charger le modèle sauvegardé s'il s'agit d'un entraînement incrémental
        if incremental_training:
            if not isinstance(new_data, list) or not all(isinstance(phrase, str) for phrase in new_data):
                raise ValueError("Les nouvelles données doivent être une liste de chaînes de caractères.")
            if not isinstance(new_labels, list):
                raise ValueError("Les nouveaux labels doivent être une liste.")

            # Encodage des nouveaux labels
            new_labels_encoded = self.label_encoder.transform(new_labels)

            # Tokenization des nouvelles données avec un padding constant 'max_length'
            new_encodings = self.tokenizer(new_data, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

            # Charger les anciennes données pour éviter l'oubli
            old_data, old_labels = self.sample_old_data(sample_size_per_label=100)
            old_labels_encoded = self.label_encoder.transform(old_labels)

            if old_data:
                # Utiliser un padding constant 'max_length' pour les anciennes données aussi
                old_encodings = self.tokenizer(old_data, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

                # Concaténer les anciennes et nouvelles données
                combined_encodings = {key: torch.cat([new_encodings[key], old_encodings[key]], dim=0) for key in new_encodings}
                combined_encoded_labels = np.concatenate([new_labels_encoded, old_labels_encoded], axis=0)

                combined_data = new_data + old_data
                combined_labels = new_labels + old_labels
            else:
                # S'il n'y a pas de données anciennes, utiliser uniquement les nouvelles
                combined_encodings = new_encodings
                combined_encoded_labels = new_labels_encoded

                combined_data = new_data 
                combined_labels = new_labels

            # Créer le dataset avec les données combinées
            train_dataset = CustomDataset(combined_encodings, combined_encoded_labels)

            # Sauvegarder les nouvelles données avec les anciennes pour les futures étapes incrémentales
            self.save_old_data(combined_data, combined_labels)

        else:
            # Si c'est un entraînement complet, utiliser toutes les données d'entraînement initiales
            train_dataset = CustomDataset(self.train_encodings, self.train_labels)

            # Sauvegarder les phrases d'entraînement brutes avant l'encodage pour un futur entraînement incrémental
            train_phrases = self.data_preprocessor.decode_data(self.train_encodings)  # Assurez-vous de sauvegarder les phrases brutes
            train_labels = self.label_encoder.inverse_transform(self.train_labels).tolist()
            self.save_old_data(train_phrases, train_labels)


        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.02,
            logging_dir='./logs',
            logging_steps=50,
            save_steps=500,
            save_total_limit=3,
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            gradient_accumulation_steps=2,
            fp16=True,
            learning_rate=1e-5,
            lr_scheduler_type="cosine_with_restarts",
            save_strategy="epoch",
            report_to="none",
            dataloader_num_workers=4,
        )

        # Vérification des données de validation
        if hasattr(self, 'val_encodings') and hasattr(self, 'val_labels'):
            val_dataset = CustomDataset(self.val_encodings, self.val_labels)
        else:
            val_dataset = None

        # Utilisation du CustomTrainer sans passer 'loss_strategy' directement au constructeur
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if val_dataset else train_dataset,
            loss_strategy=self.loss_strategy,
            compute_metrics=self.metrics_strategy.compute_metrics
        )

        # Entraîner le modèle
        trainer.train()

        # Sauvegarder le modèle après l'entraînement
        self.save_model(self.model_dir)



    def save_model(self, model_dir):
        """
        Sauvegarde le modèle, le tokenizer et le label encoder.

        Args:
            model_dir (str): Le répertoire où sauvegarder les fichiers.
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Sauvegarder le modèle et le tokenizer
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        # Sauvegarder le label_encoder
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)


    def load_model(self, model_dir):
        """
        Charge le modèle, le tokenizer et le label encoder à partir des fichiers sauvegardés.

        Args:
            model_dir (str): Le répertoire contenant les fichiers sauvegardés.
        """
        # Charger le modèle et le tokenizer
        self.tokenizer = DebertaTokenizer.from_pretrained(model_dir)
        self.model = DebertaForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model_dir = model_dir

        # Charger le label_encoder
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)


    def predict_intent(self, text, threshold=0.7):
        """
        Prédit l'intention d'une phrase donnée avec un seuil de confiance.

        Args:
            text (str): La phrase à prédire.
            threshold (float, optionnel): Le seuil de confiance pour accepter une prédiction. Défaut: 0.7

        Returns:
            str: L'intention prédite ou 'unknown' si la confiance est inférieure au seuil.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            max_prob, predicted_label = torch.max(probabilities, dim=1)

            if max_prob < threshold:
                return "unknown"
            else:
                try:
                    return self.label_encoder.inverse_transform(predicted_label.cpu().numpy())[0]
                except ValueError:
                    return "unknown"
