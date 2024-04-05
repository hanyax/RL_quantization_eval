# Disable CUDA
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='-1'

import torch
from torch.quantization import quantize_dynamic
from torch.quantization.observer import HistogramObserver, MinMaxObserver
from torch import nn

import gym_oscillator
#import oscillator_cpp
import gym
# from stable_baselines_master.stable_baselines import PPO2
from stable_baselines3 import PPO
from stable_baselines3 import SAC
# from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import scipy.stats as ss
import scipy
import matplotlib.pyplot as plt

from genesys_quantized_linear import quantize
from sac_python_model import qSAC

eval = False
quant = not eval
runPPO = False
runSAC = not runPPO
n_timesteps = 5000

if runPPO:
    model_path = "/home/hanyang/RL_quantization_eval/rl-trained-agents/testPPO_exp_1000_pytorch_test50"
    model = PPO.load(model_path)
    model_q = PPO.load(model_path)

elif runSAC:
    model_path = "/home/hanyang/RL_quantization_eval/rl-trained-agents/testSAC_exp_1000_pytorch_test50_lr5e_4"
    model = SAC.load(model_path)
    model_q = SAC.load(model_path)

print(model_q.policy)

env_id = 'oscillator-v0'
env = gym.make(env_id)
obs = env.reset()

# Obatain the actor network
####################################################
####################################################
if quant:
    if runPPO:
        policy_net = model_q.policy.mlp_extractor.policy_net
        action_net = model_q.policy.action_net
    elif runSAC:
        policy_net = model_q.policy.actor.latent_pi
        mu_net = model_q.policy.actor.mu
        prob_net = model_q.policy.actor.log_std

    policy_net.eval()

    if runPPO:
        action_net.eval()
    elif runSAC:
        mu_net.eval()
        prob_net.eval()

    # print("Full Precision Network Structure")
    # print(policy_net)
    # print(action_net)

    # # Full precision weights
    # ####################################################
    # ####################################################
    weight1_float = policy_net[0].weight.detach().numpy()
    bias1_float = policy_net[0].bias.detach().numpy()
    weight2_float = policy_net[2].weight.detach().numpy()
    bias2_float = policy_net[2].bias.detach().numpy()

    # # Quantization
    # ####################################################
    # ####################################################

    # # Dynamic Quantization
    # # policy_net = quantize_dynamic(
    # #     model=policy_net, qconfig_spec={nn.Linear}, dtype=torch.qint8, inplace=False
    # # )

    # # action_net = quantize_dynamic(
    # #     model=action_net, qconfig_spec={nn.Linear}, dtype=torch.qint8, inplace=False
    # # )

    # # # Static Quantization
    # ####################################################
    # ####################################################

    policy_net = nn.Sequential(torch.quantization.QuantStub(), 
                    *policy_net, 
                    torch.quantization.DeQuantStub())

    if runPPO:
        action_net = nn.Sequential(torch.quantization.QuantStub(), 
                        action_net, 
                        torch.quantization.DeQuantStub())
    # elif runSAC:
        # mu_net = nn.Sequential(torch.quantization.QuantStub(), 
        #                 mu_net, 
        #                 torch.quantization.DeQuantStub())
        
        # prob_net = nn.Sequential(torch.quantization.QuantStub(), 
        #                 prob_net, 
        #                 torch.quantization.DeQuantStub())

    # # """Prepare"""

    # # Default to per channel with higher accuracy 
    # ##########################
    # ##########################
    # # backend = "x86"
    # # policy_net.qconfig = torch.quantization.get_default_qconfig(backend)

    # # GeneSys can not do per channel yet, use per tensor for now
    # ##########################
    # ##########################
    qconfig1 = torch.quantization.QConfig(
        activation=HistogramObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
        weight=HistogramObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.qint8)
    )

    qconfig2 = torch.quantization.QConfig(
        activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
        weight=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.qint8)
    )

    policy_net.qconfig = qconfig2

    if runPPO:
        action_net.qconfig = qconfig2
    # elif runSAC:
    #     mu_net.qconfig = qconfig1
    #     prob_net.qconfig = qconfig2

    torch.quantization.prepare(policy_net, inplace=True)

    if runPPO:
        torch.quantization.prepare(action_net, inplace=True)
    # elif runSAC:
    #     torch.quantization.prepare(mu_net, inplace=True)
    #     torch.quantization.prepare(prob_net, inplace=True)

    print("Before Calibration: ", policy_net)
    if runPPO:
        print("Before Calibration: ", action_net)
    # elif runSAC:
    #     print("Before Calibration: ", mu_net)
    #     print("Before Calibration: ", prob_net)

    # Assign prepared network back to model to calibrate
    if runPPO:
        model_q.policy.mlp_extractor.policy_net = policy_net
        model_q.policy.action_net = action_net
    elif runSAC:
        model_q.policy.actor.latent_pi = policy_net
        # model_q.policy.actor.mu = mu_net
        # model_q.policy.actor.prob = prob_net

    # Calibration with enviorment 
    with torch.inference_mode():
        for _ in range(n_timesteps):
            action, _ = model_q.predict(obs)
            obs, reward, done, infos = env.step(action)

    print("After Calibration: ", policy_net)
    if runPPO:
        print("After Calibration: ", action_net)
    # elif runSAC:
    #     print("After Calibration: ", mu_net)
    #     print("After Calibration: ", prob_net)

    torch.quantization.convert(policy_net, inplace=True)
    if runPPO:
        torch.quantization.convert(action_net, inplace=True)
    # elif runSAC:
    #     torch.quantization.convert(mu_net, inplace=True)
    #     torch.quantization.convert(prob_net, inplace=True)

    if runPPO:
        print("Input Quantization")
        input_q = policy_net[0]
        scale_input = input_q.scale.numpy()
        zp_input =  input_q.zero_point.numpy()
        print("Input Scale: ", scale_input)
        print("Input Zero Point ", zp_input)
        print("---------------------------------------------------")

        print("Linear Layer 1")
        scale_weight1 = policy_net[1].weight().q_scale()
        zp_weight1 = policy_net[1].weight().q_zero_point()
        scale_linear1_out = policy_net[1].scale
        zp_linear1_out = policy_net[1].zero_point
        print("Linear 1 Weight Scale: ", scale_weight1)
        print("Linear 1 Weight Zero Point ", zp_weight1)
        print("Linear 1 Output Scale: ", scale_linear1_out)
        print("Linear 1 Output Zero Point ", zp_linear1_out)

        weight_q_array1 = torch.int_repr(policy_net[1].weight()).numpy().astype('int')
        zero_array1 = np.ones((64,6), dtype=int) * zp_weight1
        weight_q_array1_final = np.concatenate((weight_q_array1, zero_array1), axis=1) - zp_weight1
        np.savetxt('ppo_data/weight/sac_linear1.txt', weight_q_array1_final, fmt='%i')
        print("---------------------------------------------------")

        print("Linear Layer 2")
        scale_weight2 = policy_net[3].weight().q_scale()
        zp_weight2 = policy_net[3].weight().q_zero_point()
        scale_linear2_out = policy_net[3].scale
        zp_linear2_out = policy_net[3].zero_point
        print("Linear 2 Weight Scale: ", scale_weight2)
        print("Linear 2 Weight Zero Point ", zp_weight2)
        print("Linear 2 Output Scale: ", scale_linear2_out)
        print("Linear 2 Output Zero Point ", zp_linear2_out)
        
        weight_q_array2 = torch.int_repr(policy_net[3].weight()).numpy().astype('int32')
        print(weight_q_array2)
        np.savetxt('ppo_data/weight/sac_linear2.txt', weight_q_array2, fmt='%i')
        print("---------------------------------------------------")

        # print("Linear Layer 3")
        # print(action_net[0])
        # scale_weight3 = action_net[1].weight().q_scale()
        # zp_weight3 = action_net[1].weight().q_zero_point()
        # scale_linear3_out = action_net[1].scale
        # zp_linear3_out = action_net[1].zero_point
        # print("Linear 3 Weight Scale: ", scale_weight3)
        # print("Linear 3 Weight Zero Point ", zp_weight3)
        # print("Linear 3 Output Scale: ", scale_linear3_out)
        # print("Linear 3 Output Zero Point ", zp_linear3_out)

        # weight_q_array3 = torch.int_repr(action_net[1].weight()).numpy().astype('int32')
        # print(weight_q_array3)

        # np.savetxt('sac_data/weight/sac_linear3.txt', weight_q_array3, fmt='%i')

    elif runSAC:
        print("Input Quantization")
        input_q = policy_net[0]
        scale_input = input_q.scale.numpy()
        zp_input =  input_q.zero_point.numpy()
        print("Input Scale: ", scale_input)
        print("Input Zero Point ", zp_input)
        print("---------------------------------------------------")

        print("Linear Layer 1")
        scale_weight1 = policy_net[1].weight().q_scale()
        zp_weight1 = policy_net[1].weight().q_zero_point()
        scale_linear1_out = policy_net[1].scale
        zp_linear1_out = policy_net[1].zero_point
        scale_bias_1 = scale_input * scale_weight1
        bias1_q = quantize(src=bias1_float, scale=scale_bias_1, zero_point=0, precision=32, signed=True, isPrint=False)
        np.savetxt('sac/sac_gemm_relu1_bias.txt', bias1_q, fmt='%i')

        print("Linear 1 Weight Scale: ", scale_weight1)
        print("Linear 1 Weight Zero Point ", zp_weight1)
        print("Linear 1 Output Scale: ", scale_linear1_out)
        print("Linear 1 Output Zero Point ", zp_linear1_out)
        
        weight_q_array1 = torch.int_repr(policy_net[1].weight()).numpy().astype('int')
        zero_array1 = np.ones((256,6), dtype=int) * zp_weight1 
        weight_q_array1_final = np.concatenate((weight_q_array1, zero_array1), axis=1) - zp_weight1
        weight_q_array1_final_transposed = weight_q_array1_final.transpose()
        np.savetxt('sac/sac_gemm_relu1_weight_pre_transpose.txt', weight_q_array1_final, fmt='%i')
        np.savetxt('sac/sac_gemm_relu1_weight.txt', weight_q_array1_final_transposed, fmt='%i')
        print("---------------------------------------------------")

        print("Linear Layer 2")
        scale_weight2 = policy_net[3].weight().q_scale()
        zp_weight2 = policy_net[3].weight().q_zero_point()
        scale_linear2_out = policy_net[3].scale
        zp_linear2_out = policy_net[3].zero_point
        scale_bias_2 = scale_linear1_out * scale_weight2
        bias2_q = quantize(src=bias2_float, scale=scale_bias_2, zero_point=0, precision=32, signed=True, isPrint=False)
        np.savetxt('sac/sac_gemm_relu2_bias.txt', bias2_q, fmt='%i')

        print("Linear 2 Weight Scale: ", scale_weight2)
        print("Linear 2 Weight Zero Point ", zp_weight2)
        print("Linear 2 Output Scale: ", scale_linear2_out)
        print("Linear 2 Output Zero Point ", zp_linear2_out)
        
        weight_q_array2 = torch.int_repr(policy_net[3].weight()).numpy().astype('int') - zp_weight2
        np.savetxt('sac/sac_gemm_relu2_weight.txt', weight_q_array2, fmt='%i')
        print("---------------------------------------------------")

        # print("Linear Layer 3")
        # print(action_net[0])
        # scale_weight3 = action_net[1].weight().q_scale()
        # zp_weight3 = action_net[1].weight().q_zero_point()
        # scale_linear3_out = action_net[1].scale
        # zp_linear3_out = action_net[1].zero_point
        # print("Linear 3 Weight Scale: ", scale_weight3)
        # print("Linear 3 Weight Zero Point ", zp_weight3)
        # print("Linear 3 Output Scale: ", scale_linear3_out)
        # print("Linear 3 Output Zero Point ", zp_linear3_out)

        # weight_q_array3 = torch.int_repr(action_net[1].weight()).numpy().astype('int32')
        # print(weight_q_array3)

        weight_q_array3 = np.ones((256,16), dtype=int)
        bias_q_array3 = np.ones((16), dtype=int)
        np.savetxt('sac/sac_gemm3_weight.txt', weight_q_array3, fmt='%i')
        np.savetxt('sac/sac_gemm3_bias.txt', bias_q_array3, fmt='%i')

        weight_q_array4 = np.ones((256,16), dtype=int)
        bias_q_array4 = np.ones((16), dtype=int)
        np.savetxt('sac/sac_gemm4_weight.txt', weight_q_array4, fmt='%i')
        np.savetxt('sac/sac_gemm4__bias.txt', bias_q_array4, fmt='%i')

    # # Single Inference
    # ##########################################
    # ##########################################
    obs = env.reset()  
    action, _states = model.predict(obs)
    # Input Quantization 
    input_fp = obs
    input_q = quantize(obs, zero_point=0, scale=scale_input)
    input_pad = np.zeros((6), dtype=int)
    input_pad_final = np.concatenate((input_q, input_pad), axis=0)
    input_final = np.zeros((2,256))
    input_final[0] = input_pad_final
    input_final[1] = input_pad_final
    input_final = input_final.flatten()
    np.savetxt('sac/sac_gemm_relu1_data.txt', input_pad_final, fmt='%i')

    # qSAC:
    # print("Input FP:")
    # print(input)
    # print("Weight FP:")
    # print(weight1_float)
    # print("Bias FP:")
    # print(bias1_float)

    quantized_SAC = qSAC(weight1_fp = weight1_float, weight1_fxp=weight_q_array1_final_transposed,
                        bias1_fp=bias1_float, bias1_fxp = bias1_q,
                        M_1=scale_bias_1)

    print("Quantized Forward")
    out_int_linear = quantized_SAC.forward_int(input_pad_final) 
    print("Quantized Forward Float Output")
    print(out_int_linear)
    out_int_re_quant = quantize(src=out_int_linear, scale=scale_linear1_out, zero_point=0, signed=False)
    print("Gemm Requantize Scale", 1/scale_linear1_out)
    np.savetxt('sac/sac_output_golden.txt', out_int_re_quant, fmt='%i')
    print("Quantized Forward Int Output")
    print(out_int_re_quant)


    print("Fp Forward")
    out_float = quantized_SAC.forward_float(input_fp)
    print(out_float)

    print("diff")
    error_rate = np.average(np.abs(out_int_linear-out_float)) / np.average(np.abs(out_float))
    print(error_rate)

# Evaluation
##########################
##########################

if eval:
    #Store rewards
    rews_ = []
    #Store observations
    obs_ = []
    obs = env.reset()
    #Store actions
    acs_ = []
    #Store X,Y according to 
    states_x = []
    states_y = []

    #Initial, non-suppresssion 
    TT=7500
    for i in range(TT):
        obs, rewards, dones, info = env.step([0])
        
        states_x.append(env.x_val)
        states_y.append(env.y_val)
        obs_.append(obs[0])
        acs_.append(0)
        # rews_.append(rewards)

    #Suppression stage
    for i in range(TT):    
        action, _states = model.predict(obs)
        #action, _states = model_q.predict(obs)
        obs, rewards, dones, info = env.step(action)

        states_x.append(env.x_val)
        states_y.append(env.y_val)
        obs_.append(obs[0])
        acs_.append(action[0])
        rews_.append(rewards)

    #Final relaxation
    for i in range(5000):
        obs, rewards, dones, info = env.step([0])
        states_x.append(env.x_val)
        states_y.append(env.y_val)
        obs_.append(obs[0])
        acs_.append(0)
        # rews_.append(rewards)


    print(np.average(rews_))

    ######################################################################33    
        
    '''state action plot '''
    plt.figure(figsize=(12,3))   
    ax = plt.subplot()  
    # ax.spines["top"].set_visible(False)  
    # ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18) 
    # plt.figure(figsize=(12,3))    
    plt.plot(states_x,lw=2, color = 'darkslateblue')
    ax.set_ylabel('X(t)', fontsize=20, color = 'darkslateblue') 
    ax.tick_params(axis='y')
    ax2 = ax.twinx() 
    ax2.set_ylabel('Action', fontsize=20, color='lightseagreen') 
    plt.plot(acs_, color='lightseagreen', lw=2)
    plt.yticks(fontsize= 18)
    ax2.set_ylim(-1.0,2.5)
    if runPPO:
        plt.title('Synchrony Supression of SAC', fontsize = 26)
        plt.savefig('figure/sac_fp.png')
    elif runSAC:
        plt.title('Synchrony Supression of SAC', fontsize = 26)
        plt.savefig('figure/sac_fp.png')