from torch.utils.data import Dataset, DataLoader

class testSet(Dataset):
    def __init__(self, n):
        self.data = [i for i in range(n)]
        self.labels =  [i for i in range(n)]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)
