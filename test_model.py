from data_feeder import SlidingWindowDataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

import torch.nn as nn
import matplotlib.pyplot as plt


class Tester:
    def __init__(self, model, model_state_dir, test_csv_dir, appliance, dataset):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.criterion = nn.MSELoss()

        # set up the model and its parameters
        self.model = model
        checkpoint = torch.load(model_state_dir, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # parameters for normalisation
        self.aggregate_mean = checkpoint['aggregate_mean']
        self.aggregate_std = checkpoint['aggregate_std']
        self.appliance_mean = checkpoint['appliance_mean']
        self.appliance_std = checkpoint['appliance_std']

        self.appliance = appliance
        self.appliance_name_formatted = self.appliance.replace("_", " ")
        self.dataset = dataset

        # set up the dataloader
        self.offset = 299
        self.batch_size = 1000
        test_dataset = SlidingWindowDataset(test_csv_dir, offset=self.offset)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # set up a dataframe for the results
        test_df = pd.read_csv(test_csv_dir, low_memory=False)
        self.timestamps = test_df["time"].iloc[self.offset:-self.offset].reset_index(drop=True)
        self.predictions = []
        self.ground_truth = []
        self.aggregate = []

    def testModel(self):
        """
        Test the model on the test dataset and collect predictions.
        """
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
                for inputs, targets in self.test_loader:
                        inputs_normalised = (inputs - self.aggregate_mean) / self.aggregate_std
                        targets_normalised = (targets - self.appliance_mean) / self.appliance_std
                        inputs_normalised, targets_normalised = inputs_normalised.to(self.device), targets_normalised.to(self.device)
                        outputs = self.model(inputs_normalised)
                        loss = self.criterion(outputs.squeeze(-1), targets_normalised)
                        test_loss += loss.item()

                        denormalised_outputs = outputs.squeeze(-1) * self.appliance_std + self.appliance_mean
                        denormalised_targets = targets_normalised * self.appliance_std + self.appliance_mean
                        denormalised_inputs = inputs_normalised * self.aggregate_std + self.aggregate_mean

                        self.predictions.extend(denormalised_outputs.cpu().numpy().flatten())
                        self.ground_truth.extend(denormalised_targets.cpu().numpy().flatten())
                        self.aggregate.extend(denormalised_inputs[:, self.offset].cpu().numpy().flatten()) # use the center of the window for the aggregate

        # match timestamp length to number of predictions
        trim_length = len(self.predictions)
        self.timestamps = self.timestamps[:trim_length]

        # remove values where prediction < 0
        self.predictions = [max(0, pred) for pred in self.predictions]
        self.ground_truth = [max(0, gt) for gt in self.ground_truth]
        self.aggregate = [max(0, agg) for agg in self.aggregate]

        test_loss /= len(self.test_loader)
        print(f"Test Loss: {test_loss}")
    
    def getResults(self):
        """
        Return the results of the test as a pandas dataframe
        """
        results_df = pd.DataFrame({
                "time": self.timestamps,
                "aggregate": self.aggregate,
                "prediction": self.predictions,
                "ground truth": self.ground_truth
        })
        return results_df

    def getMetrics(self):
        """
        Calculate the metrics for the test.
        """
        MAE = np.mean(np.abs(np.array(self.predictions) - np.array(self.ground_truth)))
        MSE = np.mean((np.array(self.predictions) - np.array(self.ground_truth)) ** 2)
        return MAE, MSE

    def plotResults(self):
        """
        Plot the results of the test.
        """
        results_df = pd.DataFrame({
                "time": self.timestamps,
                "aggregate": self.aggregate,
                "prediction": self.predictions,
                "ground truth": self.ground_truth
        })
        
        # Plot the results
        plt.figure(figsize=(30, 6))
        results_df["time"] = pd.to_datetime(results_df["time"])
        plt.plot(results_df["time"], results_df["aggregate"], label="Aggregate", alpha=0.7)
        plt.plot(results_df["time"], results_df["ground truth"], label="Ground Truth", alpha=0.7)
        plt.plot(results_df["time"], results_df["prediction"], label="Prediction", alpha=0.7)
        plt.xlabel("Timestamp")
        plt.ylabel("Power (Normalized)")
        plt.legend()
        plt.title("Aggregate, Ground Truth, and Prediction Comparison")
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to fit rotated labels
        plt.show()