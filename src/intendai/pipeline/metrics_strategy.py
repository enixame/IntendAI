# src/intendai/pipeline/metrics_strategy.py

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class MetricsStrategy:
    """
    Interface pour les stratégies de calcul des métriques.
    """
    def compute_metrics(self, pred):
        raise NotImplementedError("This method should be overridden.")

class WeightedMetrics(MetricsStrategy):
    """
    Stratégie de calcul des métriques pondérées.
    """
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}
