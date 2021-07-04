import os
import time
import neptune
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from topology import TopologyDriver
from util_heo import timeSince, training_manager, open_log, weights_initializer
from model import Call_Model



def ActionSample(node_logits, vnf_logits, mask, epsilon):
    '''
    B is the batch_size
    N is the number of nodes in the topology

    - INPUT
    node_logits : <B, N>    Tensor, final logits of node actions
    vnf_logits  : <B, N, 2> Tensor, final logits of vnf processing actions
    mask        : <B, N> int Array, binary mask for indicate valid node actions
    epsilon     : float           , probability for exploration (0 means greedy sampling)

    - OUTPUT
    action_logprob : <B>    Tensor, log probability of sampled action
    node_action    : <B> int Array, indexes of generated node actions
    vnf_action     : <B> int Array, indexes of generated vnf processing actions

    '''
    softmax = nn.Softmax(dim=1)

    B = len(TDset)

    exploration = 1 if random.random() < epsilon else 0

    action_logprob = None
    node_action = np.zeros(B)
    vnf_action = np.zeros(B)

    node_probs = softmax(node_logits)
    node_pred = torch.argmax(node_probs, dim=1)

    for b in range(B):
        if exploration == 1:
            tmp_node_action = random.choice([i for i, val in enumerate(mask) if val > 0])
            tmp_vnf_action = random.choice([0,1])
        else:
            tmp_node_action = node_pred[b].item()
            tmp_vnf_action = torch.argmax(vnf_logits[b, node_pred[b], :].unsqueeze(0), dim=1)
        
        vnf_probs = softmax(vnf_logits[b, node_pred[b], :].unsqueeze(0))
        tmp_logprob = torch.log(node_probs[b,tmp_node_action] *\
                                 vnf_probs[0,tmp_vnf_action] + 1e-8).unsqueeze(0)

        action_logprob = tmp_logprob if action_logprob is None else\
                            torch.cat((action_logprob, tmp_logprob), 0)
        node_action[b] = tmp_node_action
        vnf_action[b] = tmp_vnf_action

    return action_logprob, node_action, vnf_action

def ComputeRewards(TDset, req_step, delay_coeff, reward_mode):
    '''
    B is the batch_size
    
    - INPUT
    TDset       : <B> class TD, TopologyDrivers
    req_step    : int         , time index of the current request in request sequences
    delay_coeff : float       , the controlling parameter of delay reward
    reward_mode : str         , reward type ('REINFORCE', )

    - OUTPUT
    rewards : <B> int Array, set of rewards

    '''
    B = len(TDset)

    rewards = np.zeros(B)

    for b in range(B):
        TD = TDset[b]
        req_idx = list(TD.reqs.keys())[req_step]
        
        rewards[b] = TD.ComputeReward(req_idx, delay_coeff, reward_mode) 

    return rewards

def REINFORCE_train(TDset, model, optimizer, data_spec, mode, device, max_gen,\
                     epsilon, delay_coeff, discount_factor):
    '''
    B is the batch_size
    N is the number of nodes in the topology

    - INPUT
    TDset           : <B> class TD, TopologyDrivers which is already set 
                               with requests, deployments, and labels
    model           : class model
    optimizer       : class optimizer
    data_spec       : { 'max_reqs'         : MR,
                        'max_depls'        : MD,
                        'max_labels'       : ML,
                        'n_req_features'   : FR,
                        'n_depl_features'  : FD,
                        'n_label_features' : FL }
    mode            : str          , 'NodeLevel' or 'VNFLevel'
    device          : str          , device that the model is running on
    max_gen         : int          , maximum generation step
    epsilon         : float        , probability for exploration
    delay_coeff     : float        , the controlling parameter of delay reward
    discount_factor : float        , discount factor for computaton of returns

    '''

    model.train()
    softmax = nn.Softmax(dim=1)
    
    B = len(TDset)
    N = TDset[0].n_nodes

    total_loss = 0
    n_predictions = 0
    for r_step in range(data_spec['max_reqs']):
        annotation, A_out, A_in, enc_mask, training_flag = MakeEncodingBatch(TDset, r_step)
        if training_flag == False:
            break

        # Encoding Stage
        enc_out = model.encoder(annotation, A_out, A_in)

        # Decoding Stage
        rewards = None
        saved_log_probs = None
        hidden = None
        for gen_step in range(max_gen):
            from_node, vnf_now, vnf_all, dec_mask, training_flag\
                                        = MakeDecodingBatch(TDset, r_step, mode=mode)
            if training_flag == False:
                break

            mask = enc_mask.reshape(B,1)*dec_mask
            sample_mask = [1 if sum(vals) > 0 else 0 for vals in mask]
            sample_mask_tensor = Variable(torch.from_numpy(sample_mask)).to(device)

            n_predictions += np.sum(sample_mask)

            # Run decoder
            node_logits, vnf_logits, hidden = model(enc_out, from_node,\
                                                    vnf_now, vnf_all, mask, hidden)

            # Choose actions
            action_logprob, node_action, vnf_action =\
                                ActionSample(node_logits, vnf_logits, mask, epsilon)

            # Update Generations
            UpdateGenerations(TDset, r_step, node_action, vnf_action)

            # Save transition (logprob, reward)
            action_logprob = (sample_mask_tensor*action_logprob).unsqueeze(1)

            saved_log_probs = action_logprob if saved_log_probs is None else
                                torch.cat((saved_log_probs, action_logprob), 1)

            action_reward = ComputeRewards(TDset, r_step, delay_coeff, reward_mode='REINFORCE')
            action_reward = (sample_mask*action_reward).reshape(B,1)

            rewards = action_reward if rewards is None else
                                np.concatenate((rewards, action_reward), 1)

        # Compute Returns
        returns = None
        n_transitions = rewards.shape[1]

        R = np.zeros((B,1))
        for i in range(n_transitions)[::-1]:
            reward = rewards[:,i].reshape(B,1)
            R = reward + discount_factor*R
            returns = R if returns is None else np.concatenate((R, returns),1)
    
        for b in range(B):
            returns[b] = (returns[b] - np.mean(returns[b])) / (np.std(returns[b]) + 1e-8)
        returns = Variable(torch.from_numpy(returns)).to(device)

        train_loss = 0
        for i in range(n_transitions):
            logprob = saved_log_probs[i,:]
            R = returns[i,:]
            train_loss += (-logprob*R).sum()

        train_loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 2)
        optimizer.step()
        optimizer.zero_grad()

        # Update Capacities
        UpdateCapacities(TDset, r_step)

        total_loss += train_loss.item()

    return total_loss, n_predictions

