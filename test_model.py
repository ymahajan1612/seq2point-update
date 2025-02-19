from data_feeder import SlidingWindowDataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from seq2Point_factory import Seq2PointFactory
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import re
import json
import time 

class Tester:
    def __init__(self, model_state_dir, test_csv_dir, appliance, normalisation_parameters_dir):
        """
        Tester class for testing the model
        model_name (str): Name of the model to test.
        model_state_dir (str): Directory to load the model state from.
        test_csv_dir (str): Directory to load the test CSV from.
        appliance (str): Name of the appliance to test the model for.
        normalisation_parameters_dir (str): Directory to load the normalisation parameters from.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.criterion = nn.MSELoss()
        

        # set up the model and its parameters
        checkpoint = torch.load(model_state_dir, map_location=self.device)
        self.model_name = checkpoint['model_name']
        window_length = checkpoint['window_length']
        self.model = Seq2PointFactory.createModel(self.model_name, window_length)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # extract the normalisation parameters from the house
        self.appliance_name_formatted = appliance.replace(" ", "_")
        # get the house number from the test csv directory
        file_name = os.path.basename(test_csv_dir)
        house_num = re.search(r'H(\d+)', file_name).group(1)

        with open(normalisation_parameters_dir, 'r') as f:
            self.normalisation_params = json.load(f)
        self.aggregate_mean = self.normalisation_params[house_num]["aggregate_mean"]
        self.aggregate_std = self.normalisation_params[house_num]["aggregate_std"]
        self.appliance_mean = self.normalisation_params[house_num][f'{self.appliance_name_formatted}_mean']
        self.appliance_std = self.normalisation_params[house_num][f'{self.appliance_name_formatted}_std']

        # set up the dataloader
        self.batch_size = 32
        self.offset = int((0.5 * window_length) - 1)
        test_dataset = SlidingWindowDataset([test_csv_dir], self.model.getWindowSize())
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # set up a dataframe for the results
        test_df = pd.read_csv(test_csv_dir, low_memory=False)
        self.dt = 0
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
        time_start = time.time()
        with torch.no_grad():
                for inputs, targets in self.test_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs.squeeze(-1), targets)
                        test_loss += loss.item()

                        denormalised_outputs = outputs.squeeze(-1) * self.appliance_std + self.appliance_mean
                        denormalised_targets = targets * self.appliance_std + self.appliance_mean
                        denormalised_inputs = inputs * self.aggregate_std + self.aggregate_mean

                        self.predictions.extend(denormalised_outputs.cpu().numpy().flatten())
                        self.ground_truth.extend(denormalised_targets.cpu().numpy().flatten())
                        self.aggregate.extend(denormalised_inputs[:, self.offset].cpu().numpy().flatten()) # use the center of the window for the aggregate


        # match timestamp length to number of predictions
        trim_length = len(self.predictions)
        self.timestamps = self.timestamps[:trim_length]

        # set predictions so they are greater than 0 and less than the aggregate
        self.predictions = [max(0, pred) for pred in self.predictions]
        self.predictions = [min(pred, agg) for pred, agg in zip(self.predictions, self.aggregate)]
        self.ground_truth = [max(0, gt) for gt in self.ground_truth]
        self.aggregate = [max(0, agg) for agg in self.aggregate]

        test_loss /= len(self.test_loader)
        time_end = time.time()
        self.dt = time_end - time_start
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
        Also return the time taken for disaggreation.
        """
        # calculate mean absolute error
        MAE = np.mean(np.abs(np.array(self.predictions) - np.array(self.ground_truth)))
        
        # calculate signal aggregate error
        SAE = abs(sum(self.predictions) - sum(self.ground_truth))/sum(self.ground_truth)

        return MAE, SAE, self.dt

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
        plt.ylabel("Power (Watts)")
        plt.legend()
        plt.title("Aggregate, Ground Truth, and Prediction Comparison")
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to fit rotated labels
        plt.show()