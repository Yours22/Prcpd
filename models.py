import torch
import torch.nn as nn

class POD_DNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=200):
        super(POD_DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)