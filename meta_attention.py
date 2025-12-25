import torch
import torch.nn as nn
import math

class MetaAttention(nn.Module):
    """
    MetaAttention attends over reasoning paths instead of tokens.
    A global query is formed from the model's internal state.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, H):
        """
        H: [batch, seq_len, d_model]
        """
        # Global reasoning query
        Q = self.q(H.mean(dim=1, keepdim=True))
        K = self.k(H)
        V = self.v(H)

        attn = torch.softmax(
            Q @ K.transpose(-2, -1) / math.sqrt(H.size(-1)),
            dim=-1
        )

        return attn @ V
