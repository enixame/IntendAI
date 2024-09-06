from transformers import DebertaTokenizer, DebertaForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
from sklearn.utils import resample

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        return len(self.labels)

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def balance_data(data, labels):
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

class IntentPredictionPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        self.model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels=7)
        self.model.to(self.device)

        all_possible_labels = ['greetings', 'health_status', 'backseat', 'bad', 'common', 'common_confirmation', 'ask']
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_possible_labels)

    def add_training_data(self, data, labels):
        # Encoder les labels en nombres entiers
        labels = self.label_encoder.transform(labels)
        
        # Rééchantillonnage pour équilibrer les classes sous-représentées
        data_balanced, labels_balanced = balance_data(data, labels)

        # Diviser en jeux de données d'entraînement et de validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            data_balanced, labels_balanced, test_size=0.2, random_state=42
        )

        # Encoder les phrases avec le tokenizer DeBERTa
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

        self.train_encodings = train_encodings
        self.train_labels = train_labels
        self.val_encodings = val_encodings
        self.val_labels = val_labels

        # Calculer les poids de classe pour compenser le déséquilibre
        unique_classes = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels)

        # Ajuster les poids en fonction des classes présentes
        weights = np.ones(self.model.num_labels, dtype=np.float32)
        for i, class_idx in enumerate(unique_classes):
            weights[class_idx] = class_weights[i]

        self.class_weights = torch.tensor(weights, dtype=torch.float).to(self.device)

    def train_model(self):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=20,  # Augmenter le nombre d'époques
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
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
            learning_rate=1e-5,  # Diminuer le taux d'apprentissage
            lr_scheduler_type="cosine_with_restarts",
            save_strategy="epoch",
            report_to="none",
            dataloader_num_workers=4,
        )

        # Créer les datasets
        train_dataset = CustomDataset(self.train_encodings, self.train_labels)
        val_dataset = CustomDataset(self.val_encodings, self.val_labels)

        # Utiliser le trainer pour entraîner
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            class_weights=self.class_weights,
            compute_metrics=compute_metrics
        )

        trainer.train()

    def predict_intent(self, text, threshold=0.7):  # Ajuster le seuil à 0.7
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            max_prob, predicted_label = torch.max(probabilities, dim=1)

            # Ajustement du seuil
            if max_prob < threshold:
                return "unknown"
            else:
                return self.label_encoder.inverse_transform(predicted_label.cpu().numpy())[0]
