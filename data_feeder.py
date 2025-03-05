import torch
from torch.utils.data import Dataset
import pandas as pd

class SlidingWindowDataset(Dataset):
    def __init__(self, file_dirs, window_size, crop=None):
        self.window_size = window_size
        self.offset = (window_size // 2) - 1
        self.data = []  # Store each house's data separately
        self.normalisation_params = {}

        for file in file_dirs:
            house_params = {}
            df = pd.read_csv(file)
            if crop:
                df = df.head(crop)
            house_params["aggregate_mean"] = df.iloc[:, 1].mean()
            house_params["aggregate_std"] = df.iloc[:, 1].std()
            house_params["appliance_mean"] = df.iloc[:, 2].mean()
            house_params["appliance_std"] = df.iloc[:, 2].std()
            self.normalisation_params[file] = house_params
            inputs = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32)
            outputs = torch.tensor(df.iloc[:, 2].values, dtype=torch.float32)
            self.data.append((inputs, outputs, file))  # Store each house as a tuple along with the file name

    def __len__(self):
        return sum(len(inputs) - self.window_size + 1 for inputs, _ , _ in self.data)

    def getNormalisationParams(self, file_dir):
        return self.normalisation_params[file_dir]

    def __getitem__(self, idx):
        for inputs, outputs, file in self.data:
            num_windows = len(inputs) - self.window_size + 1
            if idx < num_windows:
                start_idx = idx
                end_idx = idx + self.window_size
                
                norm_params = self.getNormalisationParams(file)

                normalised_inputs = (inputs[start_idx:end_idx] - norm_params["aggregate_mean"]) / norm_params["aggregate_std"]
                normalised_outputs = (outputs[start_idx:end_idx] - norm_params["appliance_mean"]) / norm_params["appliance_std"]


                return normalised_inputs, normalised_outputs
            idx -= num_windows  # Move to the next house's data

        raise IndexError("Index out of range")
