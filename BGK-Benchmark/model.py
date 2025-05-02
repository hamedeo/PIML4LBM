# model.py
import torch
import torch.nn as nn

# D2Q9 D₈ permutations (same as in your benchmark script)
_d8_perms = [
    [0,1,2,3,4,5,6,7,8],        # I
    [0,2,3,4,1,6,7,8,5],        # r   (90°)
    [0,3,4,1,2,7,8,5,6],        # r²  (180°)
    [0,4,1,2,3,8,5,6,7],        # r³  (270°)
    [0,3,2,1,4,7,6,5,8],        # s   (mirror x)
    [0,4,3,2,1,8,7,6,5],        # r s
    [0,1,4,3,2,5,8,7,6],        # r² s
    [0,2,1,4,3,6,5,8,7],        # r³ s
]

# precompute inverse permutations
_inv_d8 = []
for perm in _d8_perms:
    inv = [perm.index(i) for i in range(9)]
    _inv_d8.append(inv)

class SymCollision(nn.Module):
    """
    A D8‐equivariant collision network:
      f_out = (1/8) sum_{g in D8} g^{-1} ○ base_net ○ g (f_in)
    """
    def __init__(self, hidden_size=50):
        super().__init__()
        self.base_net = NaiveCollision(hidden_size=hidden_size)

    def forward(self, x):
        # x shape = (batch,9)
        outs = []
        for perm, inv in zip(_d8_perms, _inv_d8):
            x_p = x[:, perm] 
            y_p = self.base_net(x_p)    
            y = y_p[:, inv]
            outs.append(y)
        # 4) average
        y_sym = torch.stack(outs, dim=0).mean(dim=0)
        return y_sym

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

