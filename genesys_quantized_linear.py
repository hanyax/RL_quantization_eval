from torch import nn
import torch
import copy
import numpy as np
import math
from fxpmath import Fxp
from fixpoint_lib import to_fix_val, to_fp, to_fix

# From https://github.com/google/gemmlowp/blob/master/doc/quantization_example.cc#L210
def quantizeMultiplierSmallerThanOne(real_multiplier):
    right_shift = 0
    real_multiplier_local = 0 + real_multiplier
    while real_multiplier_local < 0.5:
        real_multiplier_local *= 2.0
        right_shift+=1
    quantized_multiplier_fxp = Fxp(real_multiplier_local, signed=True, n_word=64, n_frac=32) # impotrant to be 64,32 for multplay reason
    return quantized_multiplier_fxp, right_shift
    
# From FBGEMM
# https://github.com/pytorch/FBGEMM/blob/8f1b8777745d412c10d254284a72d76357ac287a/include/fbgemm/QuantUtils.h#L45
def clamp(src, precision=8, signed=False):
    min_num = -(1 << (precision - 1)) if signed else 0
    max_num = ((1 << (precision - 1)) - 1) if signed else (1 << precision) - 1
    return np.minimum(np.maximum(src, min_num), max_num)

# From FBGEMM
# https://github.com/pytorch/FBGEMM/blob/8f1b8777745d412c10d254284a72d76357ac287a/include/fbgemm/QuantUtils.h#L62
def quantize(src, zero_point, scale, precision=8, signed=False, isPrint=False):
    inv_scale = 1.0/scale
    transformed_val = src * inv_scale;
    if isPrint:
        print("Input ", src)
        print("transformed_val ", transformed_val)
    # The correct way is to use a round to the nearest integer, since it is not added, use floor for now
    #transformed_val = zero_point + np.round(transformed_val)
    transformed_val = zero_point + np.round(transformed_val)
    if isPrint:
        print("transformed_val ", transformed_val)
    
    result = clamp(src=transformed_val, precision=precision, signed=signed)
    return result.astype(int)

# From FBGEMM
# https://github.com/pytorch/FBGEMM/blob/8f1b8777745d412c10d254284a72d76357ac287a/include/fbgemm/QuantUtils.h#L147
def dequantize(src, zero_point, scale):
    result = scale * (src - zero_point)
    return result

class genesys_quantized_linear(nn.Module):
    def __init__(self, weight_fxp, bias_fxp, M):
        super().__init__() 
        self.weight_fxp = weight_fxp 
        self.weight_bias = bias_fxp
        self.M = M
    
    def forward(self, input):
        # Compute Offline Weight subtract zero point
        # the forward method does not include bias handling, assuming the zero point is pre-subtracted which equals adding zero point of 0
        
        # self.input = quantize(src=input, scale=self.input_scale, zero_point=0, signed=False) # zero_point = 0 since we pre-subtract 0 point before inference

        # Gemm in integer 
        gemm_out = (input @ self.weight_fxp)

        print("Raw Int Gemm Result: ")
        print(gemm_out)
        gemm_out = gemm_out + self.weight_bias
        print("Gemm Result Adding Bias: ")
        print(gemm_out)

        # shift left by 16 since simd is in fxp not in integer 
        gemm_out_fxp: Any = np.left_shift(gemm_out, 16)

        M0, right_shift = quantizeMultiplierSmallerThanOne(self.M)
        new_gemm_out_fxp: NDArray[Any] = np.full(gemm_out_fxp.shape, fill_value = Fxp(), dtype = Fxp)
        
        print("M0 ", M0)
        print("Right Shift ", right_shift)
        # SIMD operation * M which is decompose as * M0 then right shift
        for i, val in np.ndenumerate(gemm_out_fxp):
            temp: Fxp = to_fp(val)
            temp *= to_fix(M0)
            temp >>= right_shift
            new_gemm_out_fxp[i] = temp

        print("Gemm out fxp")
        print(new_gemm_out_fxp)

        # SIMD
        # print("M in float", self.M)
        # output = gemm_out * self.M
        # M0, right_shift = quantizeMultiplierSmallerThanOne(self.M)
        # output = gemm_out * M0
        # output = output >> right_shift
        # print("M in float", self.M)
        # print("M0 in float", M0)
        # print("M0 in fxp", to_fix_val(M0))
        # print("Right shift", right_shift)
        print("Gemm Result in float: ")
        print(new_gemm_out_fxp.astype(float))
        return torch.from_numpy(new_gemm_out_fxp.astype(float))

        ########################################################
        # Original Algorithm from the paper without offline precompute
        ########################################################
        # self.input = quantize(src=input, scale=self.input_scale, zero_point=self.input_zero_point, signed=False)
        # gemm_out = (self.input @ self.weight.transpose()).astype(np.int32)
        # M = self.bias_scale/self.output_scale
        # M0, right_shift = quantizeMultiplierSmallerThanOne(M)
        
        # # pre-computed 
        # N = self.input.shape[1]
        # NZ1Z2 = N * self.input_zero_point * self.weight_zero_point

        # # SIMD reduction
        # a1 = np.sum(self.input, axis=1)
        # a2 = np.sum(self.weight, axis=1)

        # # Vector Matrix code
        # Z1a2 = (self.input_zero_point * a2).astype(np.int32)
        # Z2a1 = (self.weight_zero_point * a1).astype(np.int32)
        # for r in range(gemm_out.shape[0]):
        #     gemm_out[r,:] = gemm_out[r,:] - Z1a2
        # for c in range(gemm_out.shape[1]):
        #     gemm_out[:,c] = gemm_out[:,c] - Z2a1
        
        # # SIMD vector matrix
        # gemm_out = gemm_out + self.bias

        # # SIMD 
        # fxp_part = (M0*(NZ1Z2+gemm_out)) >> right_shift
        
        # # SIMD 
        # output = self.output_zero_point + fxp_part.astype(np.int32)

        # output_float = dequantize(src=output, scale=self.output_scale, zero_point=self.output_zero_point)
        # return output, output_float