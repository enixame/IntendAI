# src/intendai/pipeline/intent_prediction_pipeline.py

import json
import numpy as np
import os
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

        # Par défaut, utiliser les stratégies WeightedCrossEntropyLoss et WeightedMetrics
        self.loss_strategy = loss_strategy or WeightedCrossEntropyLoss(None)
        self.metrics_strategy = metrics_strategy or WeightedMetrics()

        self.label_encoder = None
        self.class_weights = None
        self.data_preprocessor = DataPreprocessor(self.tokenizer)


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


    def train_model(self, incremental=False, new_data=None, new_labels=None):
        """
        Entraîne le modèle sur les données fournies.

        Args:
            incremental (bool, optionnel): Si vrai, effectue un entraînement incrémental sur de nouvelles données.
            new_data (list, optionnel): Nouvelles phrases pour l'entraînement incrémental.
            new_labels (list, optionnel): Nouveaux labels pour l'entraînement incrémental.
        """
        incremental_training = incremental and new_data is not None and new_labels is not None
        num_train_epochs = 20 if not incremental_training else 5
        batch_size = 16 if not incremental_training else 1

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

        # Si incrémental, utiliser les nouvelles données
        if incremental_training:
            new_encodings = self.data_preprocessor.encode_data(new_data)
            train_dataset = CustomDataset(new_encodings, new_labels)
        else:
            train_dataset = CustomDataset(self.train_encodings, self.train_labels)

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

        # Affecter la stratégie de perte au CustomTrainer après l'initialisation
        trainer.loss_strategy = self.loss_strategy

        # Entraîner le modèle
        trainer.train()


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
