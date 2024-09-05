from transformers import DebertaTokenizer, DebertaForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class CustomDataset(Dataset):
    """
    Un dataset personnalisé pour gérer les encodages et les étiquettes.
    Utilisé pour fournir les données d'entraînement et de validation à PyTorch
    et à la classe Trainer de Transformers.
    """
    def __init__(self, encodings, labels):
        """
        Initialise le dataset avec des encodages et des labels.
        
        Args:
            encodings (dict): Dictionnaire d'encodages produit par le tokenizer DeBERTa.
            labels (list): Liste des labels associés aux données.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Récupère un élément du dataset à l'index `idx`.
        
        Args:
            idx (int): Index de l'élément à récupérer.
        
        Returns:
            dict: Dictionnaire contenant les encodages et le label à l'index `idx`.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).long()  # Convertir les labels en LongTensor
        return item

    def __len__(self):
        """
        Retourne la taille du dataset.
        
        Returns:
            int: Nombre total d'éléments dans le dataset.
        """
        return len(self.labels)

class CustomTrainer(Trainer):
    """
    Classe personnalisée Trainer pour utiliser une fonction de perte avec des poids de classe.
    Hérite de Trainer de Hugging Face Transformers.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        """
        Initialise CustomTrainer avec des arguments supplémentaires.
        
        Args:
            class_weights (torch.Tensor): Tenseur de poids de classe pour gérer le déséquilibre des classes.
            *args, **kwargs: Arguments standard pour Trainer.
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Calcule la fonction de perte en utilisant des poids de classe.

        Args:
            model: Le modèle utilisé pour faire des prédictions.
            inputs (dict): Dictionnaire d'inputs pour le modèle.
            return_outputs (bool): Indicateur pour retourner les outputs en plus de la perte.

        Returns:
            loss: La valeur de la perte calculée.
        """
        labels = inputs.pop("labels").long()  # Assurer que les labels sont en LongTensor
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)  # Utilise la fonction de perte avec des poids de classe
        loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class IntentPredictionPipeline:
    """
    Pipeline de prédiction d'intention utilisant le modèle DeBERTa.
    Gère le prétraitement des données, l'entraînement du modèle et la prédiction.
    """
    def __init__(self):
        """
        Initialise le pipeline avec le tokenizer DeBERTa, le modèle, et le label encoder.
        Détecte également si un GPU est disponible pour l'entraînement.
        """
        # Détection du device (GPU ou CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Chargement du tokenizer DeBERTa pré-entraîné
        self.tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        
        # Chargement du modèle DeBERTa pour la classification de séquence
        self.model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels=6)
        self.model.to(self.device)  # Déplacer le modèle sur le GPU/CPU
        
        # Initialisation d'un encodeur de labels avec toutes les classes possibles
        all_possible_labels = ['greetings', 'health_status', 'backseat', 'bad']
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_possible_labels)  # Ajuste l'encodeur avec toutes les classes possibles

    def add_training_data(self, data, labels):
        """
        Prépare les données d'entraînement et de validation en les encodant avec le tokenizer
        et en les divisant en ensembles d'entraînement et de validation.
        
        Args:
            data (list): Liste de phrases à utiliser pour l'entraînement.
            labels (list): Liste d'étiquettes correspondant aux phrases.
        """
        # Encoder les labels de texte en nombres entiers
        labels = self.label_encoder.transform(labels)
        
        # Diviser les données en ensembles d'entraînement et de validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            data, labels, test_size=0.2, random_state=42
        )

        # Encoder les phrases d'entraînement et de validation avec le tokenizer DeBERTa
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

        # Enregistrer les encodages et les labels pour l'entraînement et la validation
        self.train_encodings = train_encodings
        self.train_labels = train_labels
        self.val_encodings = val_encodings
        self.val_labels = val_labels

        # Identifier les classes présentes dans train_labels
        unique_classes = np.unique(train_labels)

        # Calculer les poids de classe uniquement pour les classes présentes
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels)

        # Initialiser un tenseur de poids avec un poids de 1.0 pour toutes les classes
        weights = np.ones(self.model.num_labels, dtype=np.float32)

        # Mettre à jour les poids pour les classes présentes
        for i, class_idx in enumerate(unique_classes):
            weights[class_idx] = class_weights[i]

        # Convertir en tenseur PyTorch et déplacer sur le GPU/CPU
        self.class_weights = torch.tensor(weights, dtype=torch.float).to(self.device)

    def train_model(self):
        """
        Configure et entraîne le modèle DeBERTa sur les données d'entraînement.
        Utilise la classe CustomTrainer pour appliquer une perte pondérée.
        """
        # Définir les arguments d'entraînement
        training_args = TrainingArguments(
            output_dir='./results',          # Répertoire de sortie pour les résultats de l'entraînement
            num_train_epochs=20,             # Nombre d'époques d'entraînement
            per_device_train_batch_size=16,  # Taille de lot par device (GPU/CPU)
            warmup_steps=1000,               # Nombre de pas d'échauffement pour le planificateur de taux d'apprentissage
            weight_decay=0.05,               # Décroissance de poids pour l'optimiseur
            logging_dir='./logs',            # Répertoire de sortie pour les journaux
            logging_steps=100,               # Fréquence d'enregistrement des journaux
            learning_rate=1e-5,              # Taux d'apprentissage initial
            fp16=True,                       # Utilisation de la précision mixte (fp16) si disponible
            evaluation_strategy="steps",     # Stratégie d'évaluation (évaluation tous les quelques steps)
            eval_steps=500,                  # Fréquence d'évaluation
            save_steps=500,                  # Fréquence de sauvegarde du modèle
            save_total_limit=2,              # Limite du nombre total de checkpoints à conserver
        )

        # Créer des datasets personnalisés pour l'entraînement et la validation
        train_dataset = CustomDataset(self.train_encodings, self.train_labels)
        val_dataset = CustomDataset(self.val_encodings, self.val_labels)

        # Utiliser CustomTrainer pour inclure la perte pondérée
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            class_weights=self.class_weights  # Passer les poids de classe au CustomTrainer
        )

        # Démarrer l'entraînement
        trainer.train()

    def predict_intent(self, text):
        """
        Prédit l'intention d'une phrase donnée en utilisant le modèle entraîné.
        
        Args:
            text (str): La phrase dont on veut prédire l'intention.
        
        Returns:
            str: L'intention prédite pour la phrase.
        """
        self.model.eval()  # Mettre le modèle en mode évaluation
        with torch.no_grad():  # Désactiver le calcul des gradients
            # Encoder la phrase d'entrée avec le tokenizer DeBERTa
            inputs = self.tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
            inputs = {key: val.to(self.device) for key, val in inputs.items()}  # Déplacer les inputs sur le GPU/CPU
            
            # Faire la prédiction
            outputs = self.model(**inputs)
            _, predicted_label = torch.max(outputs.logits, dim=1)

        # Décoder l'étiquette prédite en son format textuel
        return self.label_encoder.inverse_transform(predicted_label.cpu().numpy())[0]
