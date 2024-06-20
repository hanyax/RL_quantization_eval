# RL Model Quantization Instruction

## Overview
The script uses PyTorch Static Quantization to quantize the model parameters. It produces both the quantized model and the required quantization parameters by the NeuroWeaver accelerator.

## Step 1. Load Pre-Train PyTorch Policy
The model is loaded as a pre-trained Stable Baseline policy, an oscillator gym environment is also set up.

## Step 2. Access the Relevant Network From the Pre-train Policy
Each model has its unique inference pipeline. For example, SAC inference goes through three different network: latent_pi, mu, and log_std. Each network is extracted for quantization. 

## Step 3. Quantization Setup 
The extracted network is then packetaged with Quantization Stub which serves as both the quantization layer place holder and the calibration obverser. Quantization configurations that include a quantization scheme, calibration observer type, and quantization datatype are also set up. In the case of SAC, we use per_tensor_affine as the scheme due to NeuroWeaver design choice, QINT as input datatype, and QINT as weight data type per standard practice. 

## Step 4. Quantization Calibration
The networks with packaged Quantization Stub are then inserted back to the policy network and a calibration inference for many iterations is performed where the observer will collect the min/max of input and weight data based on this calibration run. 

## Step 5. Quantization Conversation and Parameter Extraction
After the califbration data is collected, we quantized the networks based on these calibrations. The quantized weight and bias are then stored. The quantization parameters such as scales and zero points is also displayed which need to be later entered into the NeuroWeaver Accelerator Compiler for inference on the hardware. 

## Step 6. Quantized Network Evaluation 
Finally, the quantized network is used side by side in test inference for comparison. 