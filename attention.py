import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Additive attention mechanism for sequence data.
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
    
    def forward(self, encoder_outputs):
        """
        Forward pass of the attention mechanism.
        """
        attn_scores = self.v(torch.tanh(self.attn_proj(encoder_outputs)))
        
        # Softmax the attention scores
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Calculate the context vector
        context = torch.sum(attn_weights * encoder_outputs, dim=1)
        
        return context