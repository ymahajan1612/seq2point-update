from model_pipeline.data_feeder import SlidingWindowDataset
import torch
from torch.utils.data import DataLoader
import os
import json
import pandas as pd
from model_pipeline.seq2Point_factory import Seq2PointFactory
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    def __init__(self, model_name, train_csv_dirs, validation_csv_dirs, appliance, dataset, model_save_dir, window_length=599):
        """
        Trainer class for training Seq2Point models.
        model_name (str): Name of the model to train.
        train_csv_dirs (list): List of file paths to the training CSVs.
        validation_csv_dirs (list): List of file paths to the validation CSVs.
        appliance (str): Name of the appliance to train the model for.
        dataset (str): Name of the dataset.
        model_save_dir (str): Directory to save the trained model.
        window_length (int): Length of the input window.
        """
        self.model_name = model_name
        self.model = Seq2PointFactory.createModel(self.model_name, window_length)

        # set up the loss function and optimiser
        self.appliance = appliance
        self.appliance_name_formatted = self.appliance.replace(" ", "_")
        self.dataset = dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.model_save_dir = model_save_dir

        self.criterion = nn.MSELoss()
        beta_1 = 0.9
        beta_2 = 0.999
        learning_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
        
        # Set up a LR scheduler that updates on plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2, factor=0.1)

        # create the dataloaders from the CSVs
        self.batch_size = 256
        train_dataset = SlidingWindowDataset(train_csv_dirs, self.model.getWindowSize())
        validation_dataset = SlidingWindowDataset(validation_csv_dirs, self.model.getWindowSize())
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

        # implement early stopping
        self.patience = 3
        self.best_val_loss = float("inf")
        self.min_delta = 1e-3
        self.counter = 0
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, threshold=self.min_delta)

        # store the train and val losses for plotting
        self.train_losses = []
        self.val_losses = []

    def trainModel(self, num_epochs=10):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(-1), targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in self.validation_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.squeeze(-1), targets)
                    val_loss += loss.item()

            train_loss /= len(self.train_loader)
            val_loss /= len(self.validation_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

            # Step the LR scheduler based on validation loss
            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss - self.min_delta:
                print(f"Validation loss improved from {self.best_val_loss} to {val_loss}. Saving model...")
                self.best_val_loss = val_loss
                if not os.path.exists(self.model_save_dir):
                    os.makedirs(self.model_save_dir)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'model_name' : self.model_name,
                    'window_length' : self.model.getWindowSize(),
                    'appliance' : self.appliance_name_formatted
                }, os.path.join(self.model_save_dir, f"{self.appliance}_{self.dataset}_{self.model_name}.pth"))
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered. No improvement in validation loss for {self.patience} epochs.")
                    break

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

    def plotLosses(self):
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Training and Validation Loss for {self.appliance_name_formatted} on {self.dataset} using {self.model_name}")
        plt.savefig(f'{self.appliance_name_formatted}_{self.dataset}_{self.model_name}_loss.png')
        plt.show()
