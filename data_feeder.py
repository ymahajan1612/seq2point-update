import torch
from torch.utils.data import Dataset
import pandas as pd

class SlidingWindowDataset(Dataset):
    def __init__(self, file_names, window_size, crop=None):
        self.window_size = window_size
        self.offset = (window_size // 2) - 1
        self.data = []  # Store each house's data separately

        for file_name in file_names:
            df = pd.read_csv(file_name)
            if crop:
                df = df.head(crop)
            inputs = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32)
            outputs = torch.tensor(df.iloc[:, 2].values, dtype=torch.float32)
            self.data.append((inputs, outputs))  # Store each house as a tuple

    def __len__(self):
        return sum(len(inputs) - self.window_size + 1 for inputs, _ in self.data)

    def __getitem__(self, idx):
        for inputs, outputs in self.data:
            num_windows = len(inputs) - self.window_size + 1
            if idx < num_windows:
                start_idx = idx
                end_idx = idx + self.window_size
                return inputs[start_idx:end_idx], outputs[start_idx + self.offset]
            idx -= num_windows  # Move to the next house's data

        raise IndexError("Index out of range")
