# IntendAI
IntendAI est un projet de machine learning conçu pour classer les intentions à partir de phrases en langage naturel.

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
