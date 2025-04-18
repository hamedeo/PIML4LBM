# model.py
import torch
import torch.nn as nn

class NaiveCollision(nn.Module):
    def __init__(self, hidden_size=50):
        super().__init__()
        # 9 -> hidden_size -> hidden_size -> 9
        self.network = nn.Sequential(
            nn.Linear(9, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, 9, bias=False)
	    # Do we need a ReLU again?
        )
    
    def forward(self, x):
        return self.network(x)


class MSRELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        relative_error = (input - target) / (target + self.eps)
        return torch.mean(relative_error ** 2)

