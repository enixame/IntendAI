# src/intendai/pipeline/custom_dataset.py

from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    """
    Dataset personnalisé pour le modèle Deberta avec encodage des phrases et des labels.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        if idx >= len(self.labels):
            raise IndexError(f"Index {idx} out of bounds for dataset with size {len(self.labels)}")
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        return len(self.labels)
