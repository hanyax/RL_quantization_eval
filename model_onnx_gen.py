from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


class SAC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256,256)
        self.linear2 = nn.Linear(256,256)
        self.nu = nn.Linear(256,1)
        self.prob = nn.Linear(256,1)

    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear1(x))
        nu = self.nu(x)
        prob = self.prob(x)
        return nu, prob
    

torch_model = SAC()
torch_model.eval()
torch_input = torch.randn(1, 256,requires_grad=False)
onnx_program = torch.onnx.export(torch_model, torch_input, "sac.onnx")