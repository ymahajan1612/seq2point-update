from data_feeder import SlidingWindowDataset
import torch
from torch.utils.data import DataLoader
import os
import json
import pandas as pd
from seq2Point_factory import Seq2PointFactory
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



class Trainer:
    def __init__(self, model_name, train_csv_dirs, validation_csv_dir, appliance, dataset, model_save_dir, window_length=599):
        """
        Trainer class for training Seq2Point models.
        model_name (str): Name of the model to train.
        train_csv_dirs (list): List of file paths to the training CSVs.
        validation_csv_dirs (str): file path to the validation CSV.
        appliance (str): Name of the appliance to train the model for.
        dataset (str): Name of the dataset.
        model_save_dir (str): Directory to save the trained model.
        window_length (int): Length of the input window.
        device (str): Device to train the model on (default: "cuda").
        """
        self.model_name = model_name
        self.model = Seq2PointFactory.createModel(self.model_name, window_length)

        # set up the loss function and optimiser
        self.criterion = nn.MSELoss()
        beta_1 = 0.9
        beta_2 = 0.999
        learning_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

        self.appliance = appliance
        self.appliance_name_formatted = self.appliance.replace(" ", "_")
        self.dataset = dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.model_save_dir = model_save_dir

        # create the dataloaders from the CSVs
        self.batch_size = 1000
        train_dataset = SlidingWindowDataset(train_csv_dirs, self.model.getWindowSize())
        validation_dataset = SlidingWindowDataset([validation_csv_dir], self.model.getWindowSize())
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
        train_dfs = [pd.read_csv(train_csv_dir, low_memory=False) for train_csv_dir in train_csv_dirs]
        combined_train_df = pd.concat(train_dfs, axis=0)
        self.aggregate_mean = combined_train_df["aggregate"].mean()
        self.aggregate_std = combined_train_df["aggregate"].std()
        self.appliance_mean = combined_train_df[self.appliance_name_formatted].mean()
        self.appliance_std = combined_train_df[self.appliance_name_formatted].std() 
        

    def trainModel(self, num_epochs=10):
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
                if not os.path.exists(self.model_save_dir):
                    os.makedirs(self.model_save_dir)
                torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'aggregate_mean': self.aggregate_mean,
                        'aggregate_std': self.aggregate_std,
                        'appliance_mean': self.appliance_mean,
                        'appliance_std': self.appliance_std
                }, os.path.join(self.model_save_dir,f"{self.appliance}_{self.dataset}_{self.model_name}.pth"))
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