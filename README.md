# IntendAI
IntendAI est un projet de machine learning conçu pour classer les intentions à partir de phrases en langage naturel.

## 09/09/2024 - Incremental training (entraînement incrémental)

```sample
new_labels_encoded = pipeline.label_encoder.transform(new_labels)

pipeline.train_model(incremental=True, new_data=new_phrases, new_labels=new_labels_encoded)
```

### Exemple de résultat à partir de nouvelles données:
<pre>
Phrase: 'Prends à gauche, t'as manqué quelque chose !' - Prédiction: 'unknown'
Phrase: 'Pourquoi tu refuses de prendre cette arme ? Elle est beaucoup mieux.' - Prédiction: 'ask'
Phrase: 'Non, essaie ça à la place, c'est bien plus efficace.' - Prédiction: 'common_confirmation'
Phrase: 'Tu fais n'importe quoi, c'est pas du tout la bonne façon de jouer.' - Prédiction: 'bad'
Phrase: 'Si t'avais suivi mes conseils, t'aurais déjà gagné.' - Prédiction: 'common'
Phrase: 'T'es vraiment sûr que c'est la bonne direction à prendre ?' - Prédiction: 'ask'
Phrase: 'Sérieusement, tu rates plein de trucs, explore un peu mieux !' - Prédiction: 'unknown'
{'loss': 0.9182, 'grad_norm': 0.3955821096897125, 'learning_rate': 9.200000000000001e-07, 'epoch': 1.0}
{'eval_loss': 0.7333636283874512, 'eval_accuracy': 0.86, 'eval_precision': 1.0, 'eval_recall': 0.86, 'eval_f1': 0.924731182795699, 'eval_runtime': 22.311, 'eval_samples_per_second': 4.482, 'eval_steps_per_second': 4.482, 'epoch': 1.0}
{'loss': 0.1032, 'grad_norm': 0.09518637508153915, 'learning_rate': 1.9200000000000003e-06, 'epoch': 2.0}
{'eval_loss': 0.05109911784529686, 'eval_accuracy': 0.99, 'eval_precision': 1.0, 'eval_recall': 0.99, 'eval_f1': 0.9949748743718593, 'eval_runtime': 22.6335, 'eval_samples_per_second': 4.418, 'eval_steps_per_second': 4.418, 'epoch': 2.0}
{'loss': 0.0264, 'grad_norm': 0.08039038628339767, 'learning_rate': 2.92e-06, 'epoch': 3.0}
{'eval_loss': 0.0016558794304728508, 'eval_accuracy': 1.0, 'eval_precision': 1.0, 'eval_recall': 1.0, 'eval_f1': 1.0, 'eval_runtime': 22.377, 'eval_samples_per_second': 4.469, 'eval_steps_per_second': 4.469, 'epoch': 3.0}
{'loss': 0.0021, 'grad_norm': 0.031056571751832962, 'learning_rate': 3.920000000000001e-06, 'epoch': 4.0}
{'eval_loss': 0.000971757632214576, 'eval_accuracy': 1.0, 'eval_precision': 1.0, 'eval_recall': 1.0, 'eval_f1': 1.0, 'eval_runtime': 22.333, 'eval_samples_per_second': 4.478, 'eval_steps_per_second': 4.478, 'epoch': 4.0}
{'loss': 0.0013, 'grad_norm': 0.03033999167382717, 'learning_rate': 4.92e-06, 'epoch': 5.0}
{'eval_loss': 0.0006188877741806209, 'eval_accuracy': 1.0, 'eval_precision': 1.0, 'eval_recall': 1.0, 'eval_f1': 1.0, 'eval_runtime': 22.888, 'eval_samples_per_second': 4.369, 'eval_steps_per_second': 4.369, 'epoch': 5.0}
{'train_runtime': 311.0665, 'train_samples_per_second': 1.607, 'train_steps_per_second': 0.804, 'train_loss': 0.21024172908067704, 'epoch': 5.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [05:11<00:00,  1.24s/it]
Phrase: 'Prends à gauche, t'as manqué quelque chose !' - Prédiction: 'backseat'
Phrase: 'Pourquoi tu refuses de prendre cette arme ? Elle est beaucoup mieux.' - Prédiction: 'backseat'
Phrase: 'Non, essaie ça à la place, c'est bien plus efficace.' - Prédiction: 'backseat'
Phrase: 'Tu fais n'importe quoi, c'est pas du tout la bonne façon de jouer.' - Prédiction: 'backseat'
Phrase: 'Si t'avais suivi mes conseils, t'aurais déjà gagné.' - Prédiction: 'backseat'
Phrase: 'T'es vraiment sûr que c'est la bonne direction à prendre ?' - Prédiction: 'backseat'
Phrase: 'Sérieusement, tu rates plein de trucs, explore un peu mieux !' - Prédiction: 'backseat'
</pre>

## 06/09/2024 - New Optimisations

### Optimisations incluses

- <b>Fine-tuning avancé</b> avec le modèle DeBERTa pour ajuster les poids sur les données spécifiques.
- <b>Gestion des classes non représentées</b> grâce à un rééchantillonnage et un ajustement des poids de classe.
- <b>Utilisation de poids de classe</b> pour corriger les déséquilibres dans les classes de sortie.
- <b>Entraînement prolongé</b> avec un nombre d'époques élevé et un ajustement progressif du learning rate pour de meilleures performances sur des données complexes.

<b>Durée de l'entrânement</b>: [21:14<00:00,  1.42s/it]

### Hyperparamètres utilisés

- <b>Nombre d'époques (num_train_epochs)</b> : 20
→ Cela indique que le modèle va passer 20 fois sur l'ensemble de données d'entraînement pour ajuster ses paramètres.

- <b>Taille des mini-lots (batch size)</b> :
   - <b>Entraînement</b> : 16 exemples par batch.
   - <b>Évaluation</b> : 16 exemples par batch.
   - <b>Warmup steps</b> : 500
→ Il s'agit du nombre d'étapes pendant lesquelles le learning rate augmente progressivement pour éviter des mises à jour brutales au début de l'entraînement.

- <b>Poids de décroissance (weight_decay)</b> : 0.02
→ C'est une forme de régularisation pour éviter que le modèle ne surapprenne (overfitting) sur les données d'entraînement.

- <b>Logging (logging_steps)</b> : Toutes les 50 étapes, des informations sur la progression de l'entraînement sont enregistrées.

- <b>Learning rate</b> : 1e-5
→ C'est la vitesse à laquelle le modèle ajuste ses poids pendant l'entraînement. Un taux faible (1e-5) est utilisé pour un ajustement fin.

- <b>Scheduler</b> : cosine_with_restarts
→ Ce scheduler ajuste dynamiquement le learning rate au cours de l'entraînement selon une courbe cosinusoïdale, avec des redémarrages à des moments spécifiques pour améliorer la convergence.

- <b>Évaluation à la fin de chaque époque</b> : Stratégie d'évaluation epoch pour évaluer le modèle après chaque passage complet sur les données d'entraînement.

- <b>Chargement du meilleur modèle à la fin de l'entraînement (load_best_model_at_end)</b> : Le modèle qui a eu les meilleures performances sur l'ensemble de validation est rechargé à la fin.

- <b>Métriques d'évaluation</b> :
  - Accuracy: L'exactitude est le ratio entre le nombre de prédictions correctes et le nombre total de prédictions. Si sur 100 prédictions, 90 sont correctes, l'exactitude est de 90%.
  - Precision: La précision mesure la proportion de prédictions positives correctes parmi toutes les prédictions positives faites par le modèle. Si le modèle prédit qu'une intention est correcte 50 fois, mais que seulement 40 d'entre elles sont réellement correctes, la précision est de 80% (40/50).
  - Recall: Le rappel mesure la proportion de vrais positifs détectés parmi tous les exemples positifs réels. Si le modèle détecte correctement 40 prédictions sur les 50 exemples réels positifs, le rappel est de 80% (40/50).
  - F1-score: Le F1-score est la moyenne harmonique de la précision et du rappel. Il combine les deux mesures pour obtenir un indicateur unique qui équilibre leur importance. Si un modèle a une précision de 80% et un rappel de 70%, le F1-score sera de : 0.746.

 - <b>Métriques Avancées pour la Classification</b> :
 (à réfléchir) Je suis en train de réfléchir à implémenter les métriques avancées.

   - ROC-AUC (Receiver Operating Characteristic - Area Under Curve): L'aire sous la courbe ROC mesure la capacité du modèle à distinguer entre les classes positives et négatives à différents seuils de classification.
   - PR-AUC (Precision-Recall Area Under Curve): L'aire sous la courbe précision-rappel évalue la performance du modèle en se concentrant sur la précision et le rappel à différents seuils.
   - MCC (Matthews Correlation Coefficient): Le MCC est une métrique qui prend en compte les vrais positifs, vrais négatifs, faux positifs et faux négatifs pour fournir une mesure équilibrée.
   - Cohen's Kappa: Mesure l'accord entre les prédictions du modèle et les vraies étiquettes en tenant compte de l'accord qui pourrait se produire par hasard.
   - Balanced Accuracy (Exactitude Équilibrée): Moyenne de la sensibilité (rappel) et de la spécificité.
   - Specificity (Spécificité): La proportion de vrais négatifs correctement identifiés parmi tous les négatifs réels.
   - Confusion Matrix (Matrice de Confusion): Tableau qui présente les vrais positifs, faux positifs, vrais négatifs et faux négatifs pour chaque classe.
   - Top-k Accuracy (Exactitude Top-k): La proportion de fois où la classe correcte est parmi les k classes les plus probables prédites par le modèle.
   - Mean Average Precision (MAP): Moyenne des précisions à différents points de rappel.
   - Brier Score: Mesure la précision des probabilités prédites par le modèle.

<b>Possible implémentation : </b>

<pre>
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix
)
</pre>

<pre>
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix
)

class AdvancedMetrics:
    def compute_metrics(self, pred):
        """
        Calcule des métriques avancées pour un ensemble de prédictions.
        
        Args:
            pred: Les prédictions du modèle.
        
        Returns:
            dict: Dictionnaire contenant diverses métriques avancées.
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        probs = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=1).numpy()

        # Calcul de l'exactitude
        accuracy = accuracy_score(labels, preds)

        # Précision, rappel, F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

        # ROC-AUC pour chaque classe (si binaire ou multi-class)
        try:
            roc_auc = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
        except ValueError:
            roc_auc = None  # ROC-AUC non défini pour certaines configurations

        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(labels, preds)

        # Cohen's Kappa
        kappa = cohen_kappa_score(labels, preds)

        # Matrice de Confusion
        conf_matrix = confusion_matrix(labels, preds).tolist()  # Convertir en liste pour sérialisation

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'mcc': mcc,
            'cohen_kappa': kappa,
            'confusion_matrix': conf_matrix
        }

</pre>

<pre>
# Supposons qu'on a importé AdvancedMetrics
metrics_strategy = AdvancedMetrics()

trainer = CustomTrainer(
    model=self.model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset if val_dataset else train_dataset,
    loss_strategy=self.loss_strategy,
    compute_metrics=metrics_strategy.compute_metrics  # Utiliser les métriques avancées
)
</pre>

### CPU
Le CPU est principalement utilisé pour la préparation des données, le prétraitement (tokenization, vectorization) et dans certains cas pour l'entraînement des modèles si aucune accélération GPU n'est utilisée.
- Recommandation minimale : Un processeur à 4 ou 6 cœurs, comme un Intel i5/i7 ou AMD Ryzen 5/7.
- <b>Recommandation optimale</b> : Un processeur à 8 cœurs ou plus (Intel i7/i9 ou AMD Ryzen 7/9).

### GPU
Le GPU est essentiel si vous utilisez des modèles complexes comme des réseaux de neurones profonds (transformers, BERT, GPT) ou si vous entraînez sur des ensembles de données volumineux. Le GPU accélère considérablement l'entraînement des modèles basés sur des frameworks comme TensorFlow ou PyTorch, en parallèle sur plusieurs cœurs de traitement.
- Recommandation minimale : Une carte GPU avec au moins 4 Go de VRAM (comme NVIDIA GTX 1050 Ti ou NVIDIA GTX 1650).
- <b>Recommandation optimale</b> : Un GPU plus puissant avec 8 Go ou plus de VRAM (comme NVIDIA RTX 3060, 3070 ou mieux, ou une carte dédiée à l'IA comme la NVIDIA A100).

### RAM
La RAM est utilisée pour charger les données en mémoire avant de les passer au CPU/GPU pour l'entraînement. 
- Recommandation minimale : 8 Go de RAM.
- <b>Recommandation optimale</b> : 16 Go ou plus, idéalement 32 Go pour les modèles complexes.

### Résultats

| Phrase                                                         | Prédiction            |
|----------------------------------------------------------------|-----------------------|
| 'Bonjour'                                                      | 'greetings'           |
| 'Salut ma gueule'                                              | 'greetings'           |
| 'Wesh ma gueule !'                                             | 'unknown'             |
| 'Comment ça va ?'                                              | 'health_status'       |
| 'La forme ?'                                                   | 'health_status'       |
| 'Tu devrais ralentir ici.'                                     | 'backseat'            |
| 'Tu dois aller par là.'                                        | 'backseat'            |
| 'Vas à gauche.'                                                | 'backseat'            |
| 'C'est vraiment nul ce que tu fais.'                           | 'bad'                 |
| 't'es nul.'                                                    | 'bad'                 |
| 'Aujourd'hui, il fait chaud. je suis sorti prendre un verre.'  | 'common'              |
| 'J'ai été faire quelques trucs hier, et toi tu as fait quoi ?' | 'ask'                 |
| 'c'est ça'                                                     | 'common_confirmation' |
| 'tout à fait'                                                  | 'common_confirmation' |

<b>Ask vs backseat</b>
Le modèle a une performance parfaite sur cet ensemble de test, avec une précision, un rappel, et un F1-score de 1.00 pour toutes les classes. 
Cela suggère que le modèle est capable de bien distinguer les classes ask et backseat sans erreur sur cet échantillon de données.

### Matrice de confusion :
<pre>
[[146   0]
 [  0  83]]
</pre>

- 146 instances de la classe ask ont été correctement classées.
- 83 instances de la classe backseat ont été correctement classées.
- 0 erreurs de classification dans les deux classes, c'est-à-dire qu'aucune instance n'a été mal classée.

### Rapport de classification :
|                    | precision   | recall | f1-score  | support |
|--------------------|-------------|--------|-----------|---------|
|                    |             |        |           |         |
|                ask |      1.00   |   1.00 |     1.00  |     146 |
|           backseat |      1.00   |   1.00 |     1.00  |      83 |
|                    |             |        |           |         |
|          accuracy  |             |        |     1.00  |     229 |
|          macro avg |      1.00   |   1.00 |     1.00  |     229 |
|       weighted avg |      1.00   |   1.00 |     1.00  |     229 |

<b>Classe ask :</b>
- <b>Précision (precision) :</b> 1.00 — signifie que le modèle a classé correctement toutes les instances prédites comme "ask".
- <b>Rappel (recall) :</b> 1.00 — indique que le modèle a trouvé toutes les instances "ask" dans les données.
- <b>F1-score :</b> 1.00 — une combinaison équilibrée de précision et de rappel (valeur parfaite ici).
- <b>Support :</b> 146 — c'est le nombre d'exemples "ask" dans l'ensemble de test.

<b>Classe backseat :</b>
- <b>Précision :</b> 1.00 — le modèle a classé correctement toutes les instances prédites comme "backseat".
- <b>Rappel :</b> 1.00 — toutes les instances "backseat" ont été correctement identifiées.
- <b>F1-score :</b> 1.00 — encore une fois, une performance parfaite.
- <b>Support :</b> 83 — c'est le nombre d'exemples "backseat" dans l'ensemble de test.

<b>Résumé global :</b>
- <b>Accuracy (exactitude) :</b> 1.00 — le modèle a classé correctement toutes les instances du jeu de données.
- <b>Macro avg et Weighted avg :</b> 1.00 — la moyenne des scores pour chaque classe, indiquant une performance parfaite à la fois en pondérant et sans pondérer par le nombre d'exemples dans chaque classe.

_______________________

## Structure du Projet

### 1. IntentPredictionPipeline (dans intent_prediction_pipeline.py)
- Gère le pipeline complet de prédiction d'intention : chargement des données, préparation des données, entraînement du modèle, prédiction et évaluation.
- Utilise des stratégies personnalisées pour la perte et les métriques.
- <b>Responsabilités</b> : charger les données depuis un fichier JSON, entraîner le modèle, sauvegarder/charger le modèle, et prédire l'intention à partir d'une phrase.

### 2. CustomTrainer (dans intent_prediction_pipeline.py)
- Hérite de Trainer de Hugging Face et redéfinit certaines méthodes pour personnaliser l'entraînement.
- Applique une stratégie de perte personnalisée et gère les DataLoaders avec drop_last=True pour éviter les batchs incomplets.
<b>Responsabilités</b> : gérer les DataLoaders et calculer la perte via une stratégie de perte personnalisée.

### 3. CustomDataset (dans custom_dataset.py)
- Hérite de Dataset et associe les encodages de texte aux labels pour les passer au modèle.
- <b>Responsabilités</b> : servir les données encodées et les labels pendant l'entraînement et l'évaluation.

### 4. WeightedCrossEntropyLoss (dans loss_strategy.py)
- Implémente une stratégie de perte personnalisée basée sur la fonction de perte CrossEntropyLoss de PyTorch, avec des poids de classe.
- <b>Responsabilités</b> : ajuster la perte pour tenir compte des déséquilibres entre les classes.

### 5. WeightedMetrics (dans metrics_strategy.py)
- Fournit une stratégie personnalisée pour calculer les métriques pendant l'entraînement et l'évaluation.
- <b>Responsabilités</b> : calculer les métriques comme la précision, le rappel, le F1-score et l'exactitude.

### 6. DataPreprocessor (dans data_preprocessor.py)
- Prépare les données avant l'entraînement.
- <b>Responsabilités</b> : encodage des textes avec le tokenizer et équilibrage des classes sous-représentées en utilisant des techniques de rééchantillonnage.

_______________________


## Pré-requis
- Microsoft Visual C++ 14.0 or supérieur requis. "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Installer Rust Compiler: rustup (disponible sur https://rustup.rs)
- Télécharger les pilotes NVIDIA et CUDA depuis le site officiel de NVIDIA : https://developer.nvidia.com/cuda-11-8-0-download-archive#:~:text=Click%20on%20the%20green.
- Aller sur le site officiel de PyTorch: https://pytorch.org/get-started/locally/
- Sélectionner les Options Correctes:
<pre>
   1. PyTorch Build: Stable
   2. Your OS: Sélectionne ton système d'exploitation (Linux, Windows, macOS)
   3. Package: pip
   4. Language: Python
   5. Compute Platform: CUDA 11.8 ou une version compatible
</pre>
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Installation

1. Clonez ce dépôt.
2. Installez les dépendances nécessaires avec pip:

```bash
pip install -r requirements.txt
