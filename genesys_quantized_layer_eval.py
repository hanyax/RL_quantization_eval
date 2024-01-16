from torch import nn
import torch
import numpy as np
import math
import copy
from genesys_quantized_linear import genesys_quantized_linear
from genesys_quantized_linear import quantize
from torch.quantization.observer import MovingAverageMinMaxObserver, HistogramObserver

def main():
    # Origial Model:
    model = nn.Sequential(nn.Linear(32,1))
    model.eval()

    ###############################################
    # Pytorch Quantization
    ###############################################
    m_pytorch = copy.deepcopy(model)
    m_pytorch.eval()

    """Insert stubs"""
    m_pytorch_q = nn.Sequential(torch.quantization.QuantStub(), 
                    *m_pytorch, 
                    torch.quantization.DeQuantStub())

    '''QConfig specifies static quantization observer, quantization method, and data type '''
    qconfig = torch.quantization.QConfig(
    activation=HistogramObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
    weight=HistogramObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.qint8)
    )

    # or 
    # qconfig = torch.quantization.get_default_qconfig(backend)
    # qconfig = torch.quantization.QConfig(
    #   activation=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
    #   weight=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)
    # )

    m_pytorch_q.qconfig = qconfig

    '''prepare the model'''
    torch.quantization.prepare(m_pytorch_q, inplace=True)

    '''Calibrate with Sample Data'''
    '''Use random here'''
    with torch.inference_mode():
        for _ in range(1000):
            x = torch.rand(10, 32)
            m_pytorch_q(x)
        
    """Convert"""
    torch.quantization.convert(m_pytorch_q, inplace=True)


    ###############################################
    # Custome Quantized Linear Inference
    ###############################################
    
    input_scale = m_pytorch_q[0].scale.numpy()
    input_zero_point = m_pytorch_q[0].zero_point.numpy()
    weight = model[0].weight.detach().numpy()
    weight_scale = m_pytorch_q[1].weight().q_scale()
    weight_zero_point = m_pytorch_q[1].weight().q_zero_point()
    bias = model[0].bias.detach()
    output_scale = m_pytorch_q[1].scale
    output_zero_point = m_pytorch_q[1].zero_point

    print("input_scale ", input_scale)
    print("weight_scale ", weight_scale)
    genesys_linear_layer = genesys_quantized_linear(input_scale, input_zero_point, weight, weight_scale, weight_zero_point,
                    bias, output_scale, output_zero_point)
    
    ITER=1000
    diff=[]
    out_list=[]
    count = 0
    for i in range(ITER):
        input_float = torch.rand(1,32)
        output_float_custom = genesys_linear_layer.forward(input_float.numpy())
        output_float_reference = model(input_float).detach().numpy()
        output_quantized_reference = m_pytorch_q(input_float)

        d = abs(output_float_reference-output_float_custom)
        d_percent = d/abs(output_float_reference)
        diff.append(d_percent)
        out_list.append(output_float_custom)

        if d_percent > 0.2:
            print("-----------------------Inference-----------------------")
            print("FAILED!")
            print("Abs diff: ", d)
            print("diff_percent ", d_percent)
            print("Actual Output: ", output_float_custom)
            print("Reference output: ", output_float_reference)
            print("Reference quantized output: ", output_quantized_reference)
            print("-------------------------------------------------")
            gemm = input_float @ weight.reshape(32,1)
            print("Gemm only reference: ", gemm)                                
            print("Bias float: ", bias)
            genesys_linear_layer.debug(input_float.numpy())
            #exit()
            count = count + 1
    print(count)

    ave_diff = np.average(diff)
    print("Ave diff", ave_diff)

if __name__ == "__main__":
    main()
 