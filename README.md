# IntendAI
IntendAI est un projet de machine learning conçu pour classer les intentions à partir de phrases en langage naturel.

## 06/09/2024 - New Optimisations

### Optimisations incluses

- <b>Fine-tuning avancé</b> avec le modèle DeBERTa pour ajuster les poids sur tes données spécifiques.
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
  - Accuracy
  - Precision
  - Recall
  - F1-score

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

## Structure du Projet

- **src/intendai/data/data_handler.py**: Contient la classe `DataHandler` qui gère la collecte et la gestion des données.
- **src/intendai/preprocessing/preprocessor.py**: Contient la classe `Preprocessor` pour le prétraitement du texte.
- **src/intendai/model/intent_classifier.py**: Contient la classe `IntentClassifier` pour l'entraînement et la prédiction des intentions.
- **src/intendai/pipeline/intent_prediction_pipeline.py**: Contient la classe `IntentPredictionPipeline` qui orchestre l'ensemble du processus de classification des intentions.
- **src/intendai/main.py**: Script principal pour exécuter le pipeline.
- **README.md**: Documentation du projet.
- **requirements.txt**: Liste des dépendances nécessaires pour exécuter le projet.

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
