# src/intendai/pipeline/intent_prediction_pipeline.py

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Garder les encodages et labels sur le CPU ici
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class IntentPredictionPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
        self.model.to(self.device)  # Déplacer le modèle sur le GPU/CPU
        self.label_encoder = LabelEncoder()

    def add_training_data(self, data, labels):
        # Encoder les labels de texte en nombres
        labels = self.label_encoder.fit_transform(labels)
        
        # Diviser les données en entraînement et validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            data, labels, test_size=0.2, random_state=42
        )

        # Encoder les données d'entrée avec le tokeniseur BERT
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

        # Enregistrer les encodages et les labels
        self.train_encodings = train_encodings
        self.train_labels = train_labels
        self.val_encodings = val_encodings
        self.val_labels = val_labels

    def train_model(self):
        # Configuration d'entraînement
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=10,
            per_device_train_batch_size=16,
            warmup_steps=1000,
            weight_decay=0.05,
            logging_dir='./logs',
            logging_steps=100,
            learning_rate=5e-5,
            fp16=True,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=500,
            save_total_limit=2,
        )

        # Préparer les datasets pour l'entraînement et l'évaluation
        train_dataset = CustomDataset(self.train_encodings, self.train_labels)
        val_dataset = CustomDataset(self.val_encodings, self.val_labels)

        # Trainer avec la classe Trainer de Transformers
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,  # Fournir le dataset de validation ici
        )

        trainer.train()

    def predict_intent(self, text):
        # Assure-toi que le texte est bien prétraité avant la prédiction
        inputs = self.tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Effectuer la prédiction sur le même device que le modèle
        outputs = self.model(**inputs)
        _, predicted_label = torch.max(outputs.logits, dim=1)

        # Décoder l'étiquette prédite en son format textuel
        return self.label_encoder.inverse_transform(predicted_label.cpu().numpy())[0]
