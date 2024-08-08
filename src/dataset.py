import torch
from torch.utils.data import Dataset

class PermutedDataset(Dataset):
    def __init__(self, original_dataset, num_permutations):
        self.original_dataset = original_dataset
        self.num_permutations = num_permutations
        self.num_classes = len(set([y for _, y in original_dataset]))

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        
        # Generate permuted labels, ensuring they're different from y
        y_permuted = []
        for _ in range(self.num_permutations):
            while True:
                perm_y = torch.randint(0, self.num_classes, (1,)).item() # Generate a random label(a)
                if perm_y != y:
                    y_permuted.append(perm_y)
                    break

        return x, y, *y_permuted

def collate_permuted(batch):
    x = torch.stack([item[0] for item in batch])
    y = torch.tensor([item[1] for item in batch])
    y_permuted = [torch.tensor([item[i+2] for item in batch]) for i in range(len(batch[0])-2)]
    return x, y, *y_permuted