import torch
from torch.utils.data import Dataset

class PermutedDataset(Dataset):
    def __init__(self, original_dataset, num_permutations):
        self.original_dataset = original_dataset
        self.num_permutations = num_permutations
        self.permutations = [torch.randperm(len(original_dataset)) for _ in range(num_permutations)]

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        y_permuted = [self.original_dataset[self.permutations[i][idx]][1] for i in range(self.num_permutations)]
        return x, y, *y_permuted

def collate_permuted(batch):
    x = torch.stack([item[0] for item in batch])
    y = torch.tensor([item[1] for item in batch])
    y_permuted = [torch.tensor([item[i+2] for item in batch]) for i in range(len(batch[0])-2)]
    return x, y, *y_permuted