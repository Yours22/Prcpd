import torch
import torch.nn as nn

class POD_DNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super(POD_DNN, self).__init__()

        self.activation = nn.SiLU() 
        
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.layer2(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        x = self.layer3(x)
        x = self.norm3(x)
        x = self.activation(x)
        
        x = self.layer4(x)
        x = self.norm4(x)
        x = self.activation(x)
        
        out = self.out_layer(x)
        return out