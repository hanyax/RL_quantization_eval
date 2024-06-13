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
                        weight_mu_fp, weight_mu_fxp, bias_mu_fp, bias_mu_fxp, M_mu, input_mu_scale, output_mu_zp,
                        weight_prob_fp, weight_prob_fxp, bias_prob_fp, bias_prob_fxp, M_prob, input_prob_scale, output_prob_zp):

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
        self.output_mu_zp = output_mu_zp

        self.linear_prob = genesys_quantized_linear(weight_fxp=weight_prob_fxp, bias_fxp=bias_prob_fxp, M=M_prob)
        self.weight_prob_fp = weight_prob_fp
        self.bias_prob_fp = bias_prob_fp
        self.input_prob_scale = input_prob_scale
        self.output_prob_zp = output_prob_zp

    def forward_int(self, x):
        print("Linear 1 forward")
        x = self.linear1(x)#.numpy()
        x = F.relu(x).numpy()
        # x = torch.from_numpy(x @ self.weight1_fp.transpose() + self.bias1_fp)#.numpy()
        # x = F.relu(x).numpy()
        print("---------------------------------------------------")

        # # Requantize
        x = quantize(src=x, zero_point=0, scale=self.input_2_scale, isPrint=True)

        gemm_relu2_data_final = np.zeros((2,256))
        gemm_relu2_data_final[0] = x
        gemm_relu2_data_final[1] = x
        gemm_relu2_data_final = gemm_relu2_data_final.flatten()
        np.savetxt('sac/sac_batch_size_2_gemm_relu2_data.txt', gemm_relu2_data_final, fmt='%i')        
        
        print("Linear 2 forward")
        x = self.linear2(x)#.numpy()
        x = F.relu(x).numpy()
        print(x)
        # print("---------------------------------------------------")  

        # # Uncomment when check mu and prob quantization alone
        # # x = torch.from_numpy(x @ self.weight1_fp.transpose() + self.bias1_fp)#.numpy()
        # # x = F.relu(x).numpy()
        # # x = torch.from_numpy(x @ self.weight2_fp.transpose() + self.bias2_fp)#.numpy()
        # # x = F.relu(x).numpy()

        # # Requantize: the output of the linear 2 relu is requantized by the input_mu scale 
        x_mu = quantize(src=x, zero_point=0, scale=self.input_mu_scale)
        print("linear2 output after scale")
        print(x_mu)
        x_prob = quantize(src=x, zero_point=0, scale=self.input_prob_scale) 

        mu_data_final = np.zeros((2,256))
        mu_data_final[0] = x_mu
        mu_data_final[1] = x_mu
        mu_data_final = mu_data_final.flatten()
        np.savetxt('sac/sac_batch_size_2_gemm3_data.txt', mu_data_final, fmt='%i')

        prob_data_final = np.zeros((2,256))
        prob_data_final[0] = x_prob
        prob_data_final[1] = x_prob
        prob_data_final = prob_data_final.flatten()
        np.savetxt('sac/sac_batch_size_2_gemm4_data.txt', prob_data_final, fmt='%i')

        print("Mu forward")
        mu = np.round(self.linear_mu(x_mu).numpy() + self.output_mu_zp)
        mu = np.maximum(np.minimum(mu, 255), 0)
        print("---------------------------------------------------")  
        
        print("Log Std forward")
        prob = np.round(self.linear_prob(x_prob).numpy() +  + self.output_prob_zp) 
        prob = np.maximum(np.minimum(prob, 255), 0)
        print("---------------------------------------------------")  

        np.savetxt('sac/sac_batch_size_2_mu_golden.txt', mu, fmt='%f')
        np.savetxt('sac/sac_batch_size_2_prob_golden.txt', prob, fmt='%f')

        #return x
        return mu, prob
    
    def forward_float(self, x):
        x = torch.from_numpy(x @ self.weight1_fp.transpose() + self.bias1_fp)#.numpy()
        x = F.relu(x).numpy()
        x = torch.from_numpy(x @ self.weight2_fp.transpose() + self.bias2_fp)#.numpy()
        x = F.relu(x).numpy()
        mu = torch.from_numpy(x @ self.weight_mu_fp.transpose() + self.bias_mu_fp).numpy()
        prob = torch.from_numpy(x @ self.weight_prob_fp.transpose() + self.bias_prob_fp).numpy()
        #return x
        return mu, prob
