# src/intendai/pipeline/loss_strategy.py

import torch

class LossStrategy:
    """
    Interface pour les stratégies de calcul de la perte.
    """
    def compute_loss(self, model, inputs):
        raise NotImplementedError("This method should be overridden.")

class WeightedCrossEntropyLoss(LossStrategy):
    """
    Stratégie de calcul de la perte en utilisant la CrossEntropy avec des poids de classe.
    """
    def __init__(self, class_weights):
        if class_weights is not None and class_weights.dim() == 0:
            raise ValueError("class_weights must be a 1D tensor, not a scalar")
        self.class_weights = class_weights

    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.logits

        # Vérification et ajustement des dimensions des logits et labels
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)  # Ajouter une dimension si nécessaire

        if labels.dim() == 0:
            labels = labels.unsqueeze(0)  # Ajouter une dimension si nécessaire

        # # Vérifier la forme des logits et labels avant le calcul de la perte
        # print(f"Logits shape: {logits.shape}")
        # print(f"Labels shape: {labels.shape}")
        # if self.class_weights is not None:
        #     print(f"Class weights shape: {self.class_weights.shape}")

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
        return loss


