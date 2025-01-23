import torch
from torch.utils.data import Dataset
import pandas as pd

class SlidingWindowDataset(Dataset):
    """
    PyTorch Dataset for loading sliding window data for training
    and testing seq2point models.
    """
    def __init__(self, file_name, window_size, crop=None):
        """
        PyTorch Dataset for loading sliding window data.

        Args:
        - file_name (str): Path to the CSV file.
        - offset (int): Offset to define the sliding window size.
        - crop (int, optional): Number of rows to load (default: None).
        """
        data = pd.read_csv(file_name, nrows=crop)
        self.inputs = data.iloc[:, 1].values  # Aggregate
        self.outputs = data.iloc[:, 2].values  # Appliance
        self.window_size = window_size
        self.offset = int((0.5 * window_size) - 1)
        self.num_windows = len(self.inputs) - 2 * self.offset

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        """
        Returns a sliding window and its corresponding target value.
        """
        start_idx = idx
        end_idx = idx + self.window_size
        input_window = self.inputs[start_idx:end_idx]
        target_value = self.outputs[start_idx + self.offset]
        return torch.tensor(input_window, dtype=torch.float32), torch.tensor(target_value, dtype=torch.float32)
