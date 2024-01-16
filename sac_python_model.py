from torch import nn
import torch
import numpy as np
import math
import copy
from genesys_quantized_linear import genesys_quantized_linear
from genesys_quantized_linear import quantize, dequantize
import torch.nn.functional as F

class qSAC(torch.nn.Module):
    def __init__(self, weight1_fp, weight1_fxp, bias1_fp, bias1_fxp, M_1):
        super().__init__()
        self.linear1 = genesys_quantized_linear(weight_fxp=weight1_fxp, bias_fxp=bias1_fxp, M=M_1)
        self.weight1_fp = weight1_fp
        self.bias1_fp = bias1_fp

    def forward_int(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        # dequantize
        # x = dequantize(src=x, zero_point=self.linear1.output_zero_point, scale=self.linear1.output_scale)
        return x.numpy()
    
    def forward_float(self, x):
        x = torch.from_numpy(x @ self.weight1_fp.transpose() + self.bias1_fp)
        x = F.relu(x)
        return x.numpy()