from data_feeder import SlidingWindowDataset
import torch
from torch.utils.data import DataLoader
import os
import json

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, train_csv_dir, validation_csv_dir, appliance, dataset, device="cuda"):
        self.model = model

        # set up the loss function and optimiser
        self.criterion = nn.MSELoss()
        beta_1 = 0.9
        beta_2 = 0.999
        learning_rate = 0.001
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

        self.appliance = appliance
        self.appliance_name_formatted = self.appliance.replace("_", " ")
        self.dataset = dataset
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # create the dataloaders from the CSVs
        self.offset = 299
        self.batch_size = 1000
        train_dataset = SlidingWindowDataset(train_csv_dir, self.offset)
        validation_dataset = SlidingWindowDataset(validation_csv_dir, self.offset)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

        # implement early stopping
        self.patience = 3
        self.best_val_loss = float("inf")
        self.min_delta = 1e-6
        self.counter = 0

        # store the train and val losses for plotting
        self.train_losses = []
        self.val_losses = []

        # fetch parameters for normalisation
        params_dir = os.path.join("dataset_management", f"{self.dataset}_parameters.json")
        with open(params_dir, "r") as f:
            params = json.load(f)
        self.aggregate_mean = params['aggregate']['mean']
        self.aggregate_std = params['aggregate']['std']
        self.appliance_mean = params[self.appliance_name_formatted]['mean']
        self.appliance_std = params[self.appliance_name_formatted]['std']

    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for inputs, targets in self.train_loader:
                inputs_normalised = (inputs - self.aggregate_mean) / self.aggregate_std
                targets_normalised = (targets - self.appliance_mean) / self.appliance_std
                inputs_normalised, targets_normalised = inputs_normalised.to(self.device), targets_normalised.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs_normalised)
                loss = self.criterion(outputs.squeeze(-1), targets_normalised)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in self.validation_loader:
                    inputs_normalised = (inputs - self.aggregate_mean) / self.aggregate_std
                    targets_normalised = (targets - self.appliance_mean) / self.appliance_std
                    inputs_normalised, targets_normalised = inputs_normalised.to(self.device), targets_normalised.to(self.device)

                    outputs = self.model(inputs_normalised)
                    loss = self.criterion(outputs.squeeze(-1), targets_normalised)
                    val_loss += loss.item()

            train_loss /= len(self.train_loader)
            val_loss /= len(self.validation_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

            if val_loss < self.best_val_loss - self.min_delta:
                print(f"Validation loss improved from {self.best_val_loss} to {val_loss}. Saving model...")
                self.best_val_loss = val_loss
                torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'aggregate_mean': self.aggregate_mean,
                        'aggregate_std': self.aggregate_std,
                        'appliance_mean': self.appliance_mean,
                        'appliance_std': self.appliance_std
                }, f"{self.appliance}_{self.dataset}.pth")
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered. No improvement in validation loss for {self.patience} epochs.")
                    break

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)


    def plotLosses(self, save_location=None):
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        if save_location:
            plt.savefig(save_location)