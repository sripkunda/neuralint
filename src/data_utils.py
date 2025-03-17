import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class SparseDataset(Dataset):
    def __init__(self, time_values, data_values):
        self.time_values = torch.tensor(time_values, dtype=torch.float32)  # (N, T)
        self.data_values = torch.tensor(data_values, dtype=torch.float32)  # (N, T)
        
        # Generate mask: 1 if observed, 0 if missing
        self.mask = ~torch.isnan(self.data_values)  # (N, T)
        self.time_values = torch.nan_to_num(self.time_values)
        self.data_values = torch.nan_to_num(self.data_values)
        
        # Replace NaNs in data with 0
        self.data_values[torch.isnan(self.data_values)] = 0.0

    def __len__(self):
        return len(self.data_values)

    def __getitem__(self, idx):
        return self.time_values[idx], self.data_values[idx], self.mask[idx]
    
def generate_data_from_function(f, num_samples=10000, sparsity=0.6, n_tmpts=100, noise_std=0.1):
    time_pts = np.linspace(0, 1, n_tmpts)
    X = []
    T = []
    
    for _ in range(num_samples):
        num_points = np.random.randint(int(min(sparsity * n_tmpts, n_tmpts) * 0.75), int(min(sparsity * n_tmpts, n_tmpts)))
        indices = np.sort(np.random.choice(n_tmpts, num_points, replace=False))
        
        sampled_times = time_pts[indices]
        sampled_values = f(sampled_times) + np.random.normal(0, noise_std, size=sampled_times.shape)
        
        # Initialize full arrays with NaNs
        full_values = np.full(n_tmpts, np.nan)
        full_tmpts = np.full(n_tmpts, np.nan)
        
        # Assign sampled values at the correct indices
        full_values[indices] = sampled_values
        full_tmpts[indices] = sampled_times
        
        X.append(full_values)
        T.append(full_tmpts)
    
    return np.array(T), np.array(X)

def get_dataloaders(T, X, batch_size=64, train_split=0.8, val_split=0.1):
    dataset = SparseDataset(T, X)
    
    N = len(dataset)
    train_size = int(train_split * N)
    val_size = int(val_split * N)
    test_size = N - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    dataloaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True),
        "validate": DataLoader(val_set, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False),
    }
    
    return dataloaders

def get_checkpoint_path(checkpoint_dir="model_checkpoints", checkpoint_name="best_model.pth"):
    return os.path.join(checkpoint_dir, checkpoint_name)