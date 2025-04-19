from model_pipeline.data_feeder import SlidingWindowDataset
import torch
from torch.utils.data import DataLoader
import os
from model_pipeline.seq2Point_factory import Seq2PointFactory
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Finetuner:
    def __init__(self,model_state_dir, finetune_data_dir, appliance, dataset, model_save_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # load the model
        checkpoint = torch.load(model_state_dir, map_location=self.device)
        self.model_name = checkpoint['model_name']
        self.window_length = checkpoint['window_length']
        self.model_state_dict = checkpoint['model_state_dict']
        self.model = Seq2PointFactory.createModel(self.model_name, self.window_length)
        self.model.load_state_dict(self.model_state_dict)
        self.appliance = appliance
        self.dataset = dataset 
        self.model_save_dir = model_save_dir
        
        self.freezeLayers()
        
        self.model.to(self.device)

        # set up the loss function and optimiser
        self.criterion = nn.MSELoss()
        beta_1 = 0.9
        beta_2 = 0.999
        learning_rate = 0.001
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate, betas=(beta_1, beta_2))
        # create the dataloaders from the CSVs
        self.batch_size = 1000
        finetune_dataset = SlidingWindowDataset([finetune_data_dir], self.model.getWindowSize())
        self.finetune_loader = DataLoader(finetune_dataset, batch_size=1000, shuffle=True)
        self.finetuning_losses = []
        # implement early stopping
        self.epochs = 50
        self.patience = 5
        self.best_val_loss = float("inf")
        self.min_delta = 1e-8
        self.counter = 0
    
    def freezeLayers(self):
        """
        Freezes all the layers of the model except the fully connected layers
        """
        for name, param in self.model.named_parameters():
            layer_name = name.split('.')[0]
            layer_type = dict(self.model.named_modules()).get(layer_name)
            if not isinstance(layer_type, nn.Linear):
                param.requires_grad = False
    
    def fineTune(self):
        for epoch in range(self.epochs):
            self.model.train()  
            train_loss = 0 
            for inputs,targets in self.finetune_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(-1), targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
            
            train_loss /= len(self.finetune_loader)
            print(f"Epoch {epoch+1}/{self.epochs} Loss: {train_loss}")

            if train_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = train_loss
                os.makedirs(self.model_save_dir, exist_ok=True)
                torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'model_name' : self.model_name,
                        'window_length' : self.window_length
                    }, os.path.join(self.model_save_dir,f"{self.appliance}_{self.dataset}_{self.model_name}.pth"))
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            self.finetuning_losses.append(train_loss)
    
    def plotLosses(self, save_location=None):
        """
        Plots the training and validation losses.
        with optional save location
        """
        plt.plot(self.finetuning_losses, label="Finetuning Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        if save_location:
            plt.savefig(save_location)
