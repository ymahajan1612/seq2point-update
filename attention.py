import torch
from torch import nn
from torch.nn import functional as F
class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(feature_dim, 1)
        
    def forward(self, x):
        attention_weights = self.attention_weights(x)
        attention_weights = F.softmax(attention_weights, dim=1)
        weighted_sum = torch.sum(x * attention_weights, dim=1)
        return weighted_sum