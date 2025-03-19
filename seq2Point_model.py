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
class Seq2PointReduced(Seq2PointBase):
    """
    Reduced Seq2Point model with dropout.
    The kernal size as well as the number of layers have been reduced in each layer
    """

    def __init__(self, input_window_length=599):
        super(Seq2PointReduced, self).__init__(input_window_length=input_window_length)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(8, 1), stride=(1, 1), padding='same')
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(6, 1), stride=(1, 1), padding='same')
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(5, 1), stride=(1, 1), padding='same')
        self.conv4 = nn.Conv2d(in_channels=30, out_channels=40, kernel_size=(4, 1), stride=(1, 1), padding='same')
        self.conv5 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(4, 1), stride=(1, 1), padding='same')
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(40 * self.input_window_length, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(-1)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.relu(self.conv5(x))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        
class Seq2PointCNNLSTM(Seq2PointBase):
    """
    Seq2Point model with CNN and LSTM layers.
    Single CNN layer followed by two LSTM layers.
    """

    def __init__(self, input_window_length=180):
        super(Seq2PointCNNLSTM, self).__init__(input_window_length=input_window_length)

        self.pad = nn.ConstantPad1d((1,2), 0)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, stride=1)
        self.lstm1 = nn.LSTM(input_size = 16, hidden_size = 64, batch_first = True, bidirectional = True)
        self.lstm2 = nn.LSTM(input_size = 2*64, hidden_size = 128, batch_first = True, bidirectional = True)
        self.fc1 = nn.Linear(self.input_window_length * 128 * 2,128)
        self.fc2 = nn.Linear(128,1)
        self.tanh = nn.Tanh()        

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pad(x)
        x = self.conv1(x)
        x = x.permute(0,2,1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.contiguous().view(-1, self.input_window_length * 128 * 2)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class Seq2PointBalanced(Seq2PointBase):
    """
    Balanced Seq2Point model 
    """

    def __init__(self, input_window_length=599):
        super(Seq2PointBalanced, self).__init__(input_window_length=input_window_length)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(8, 1), stride=(1, 1), padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(6, 1), stride=(1, 1), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))  

        self.conv3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(5, 1), stride=(1, 1), padding='same')

        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1))  
        self.dropout3 = nn.Dropout(0.3)


        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (self.input_window_length // 8), 768)  
        self.dropout_fc1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(768, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(-1)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)  

        x = self.relu(self.conv2(x))
        x = self.pool2(x)  

        x = self.relu(self.conv3(x))
        x = self.pool3(x)  
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout_fc1(x)

        x = self.fc2(x)
        return x
    




