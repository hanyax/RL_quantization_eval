from torch import nn
import torch
import numpy as np
import math
import copy
from genesys_quantized_linear import genesys_quantized_linear
from genesys_quantized_linear import quantize, dequantize
import torch.nn.functional as F

class qSAC(torch.nn.Module):
    def __init__(self, weight1_fp, weight1_fxp, bias1_fp, bias1_fxp, M_1,
                        weight2_fp, weight2_fxp, bias2_fp, bias2_fxp, M_2, input_2_scale,
                        weight_mu_fp, weight_mu_fxp, bias_mu_fp, bias_mu_fxp, M_mu, input_mu_scale,
                        weight_prob_fp, weight_prob_fxp, bias_prob_fp, bias_prob_fxp, M_prob, input_prob_scale):

        super().__init__()
        self.linear1 = genesys_quantized_linear(weight_fxp=weight1_fxp, bias_fxp=bias1_fxp, M=M_1)
        self.weight1_fp = weight1_fp
        self.bias1_fp = bias1_fp

        self.linear2 = genesys_quantized_linear(weight_fxp=weight2_fxp, bias_fxp=bias2_fxp, M=M_2)
        self.weight2_fp = weight2_fp
        self.bias2_fp = bias2_fp
        self.input_2_scale = input_2_scale

        self.linear_mu = genesys_quantized_linear(weight_fxp=weight_mu_fxp, bias_fxp=bias_mu_fxp, M=M_mu)
        self.weight_mu_fp = weight_mu_fp
        self.bias_mu_fp = bias_mu_fp
        self.input_mu_scale = input_mu_scale

        self.linear_prob = genesys_quantized_linear(weight_fxp=weight_prob_fxp, bias_fxp=bias_prob_fxp, M=M_prob)
        self.weight_prob_fp = weight_prob_fp
        self.bias_prob_fp = bias_prob_fp
        self.input_prob_scale = input_prob_scale

    def forward_int(self, x):
        x = self.linear1(x)#.numpy()
        x = F.relu(x).numpy()

        # Requantize
        x = quantize(src=x, zero_point=0, scale=self.input_2_scale)
        
        x = self.linear2(x)#.numpy()
        x = F.relu(x).numpy()

        # Requantize
        x_mu = quantize(src=x, zero_point=0, scale=self.input_mu_scale)
        x_prob = quantize(src=x, zero_point=0, scale=self.input_prob_scale)

        mu = self.linear_mu(x).numpy()
        prob = self.linear_prob(x).numpy()

        return prob
    
    def forward_float(self, x):
        x = torch.from_numpy(x @ self.weight1_fp.transpose() + self.bias1_fp)#.numpy()
        x = F.relu(x).numpy()
        x = torch.from_numpy(x @ self.weight2_fp.transpose() + self.bias2_fp)#.numpy()
        x = F.relu(x).numpy()
        mu = torch.from_numpy(x @ self.weight_mu_fp.transpose() + self.bias_mu_fp).numpy()
        prob = torch.from_numpy(x @ self.weight_prob_fp.transpose() + self.bias_prob_fp).numpy()
        return prob
