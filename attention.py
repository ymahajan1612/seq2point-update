import torch
import torch.nn as nn

class Attention(nn.Module):
    """
    Additive attention mechanism for sequence data.
    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.score_layer = nn.Linear(dim, 1, bias=False)  #

    def forward(self, x):
        # Compute raw  attention scores
        scores = self.score_layer(x)
        scores = scores.squeeze(-1)  

        # Normalize scores to get attention weights
        attn_weights = torch.softmax(scores, dim=1)

        attn_weights_3d = attn_weights.unsqueeze(-1)  
        reweighted_sequence = x * attn_weights_3d  

        return reweighted_sequence
