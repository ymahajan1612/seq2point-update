from abc import ABC, abstractmethod
import torch.nn as nn
import torch
import torch.nn.functional as F

class Seq2PointBase(ABC, nn.Module):
    """
    Abstract base class for Seq2Point models.
    """
    def __init__(self, input_window_length=599):
        super(Seq2PointBase, self).__init__()
        self.input_window_length = input_window_length

    def loadModel(self, file_path):
        """
        Loads the model from a file.
        """
        self.load_state_dict(torch.load(file_path))
    
    def getWindowSize(self):
        """
        Returns the input window size.
        """
        return self.input_window_length

    @abstractmethod
    def forward(self, x):
        """
        forward pass of the model.
        """
        raise NotImplementedError


class Seq2PointSimple(Seq2PointBase):
    """
    Standard Seq2Point model with 5 CNN layers.
    """

    def __init__(self, input_window_length=599):
        super(Seq2PointSimple, self).__init__(input_window_length=input_window_length)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=(10, 1), stride=(1, 1), padding='same')
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(8, 1), stride=(1, 1), padding='same')  
        self.conv3 = nn.Conv2d(in_channels=30, out_channels=40, kernel_size=(6, 1), stride=(1, 1), padding='same')
        self.conv4 = nn.Conv2d(in_channels=40, out_channels=50, kernel_size=(5, 1), stride=(1, 1), padding='same')
        self.conv5 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(5, 1), stride=(1, 1), padding='same')
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(50 * self.input_window_length, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(-1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    