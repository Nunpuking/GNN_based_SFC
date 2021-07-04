import random
import numpy as np
import torch
import torch.nn as nn

def ActionSample(node_logits, vnf_logits, mask, epsilon, B):
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

    exploration = 1 if random.random() < epsilon else 0

    action_logprob = None
    node_action = np.zeros(B, dtype=np.int)
    vnf_action = np.zeros(B, dtype=np.int)

    node_probs = softmax(node_logits)
    node_pred = torch.argmax(node_probs, dim=1)

    for b in range(B):
        if sum(mask[b]) == 0:
            tmp_node_action = 0
            tmp_vnf_action = 0
        else:
            if exploration == 1:
                tmp_node_action = random.choice([i for i, val in enumerate(mask[b]) if val > 0])
                tmp_vnf_action = random.choice([0,1])
            else:
                tmp_node_action = node_pred[b].item()
                tmp_vnf_action = torch.argmax(vnf_logits[b, node_pred[b], :].unsqueeze(0),\
                                             dim=1).item()

        vnf_probs = softmax(vnf_logits[b, node_pred[b], :].unsqueeze(0))
        tmp_logprob = torch.log(node_probs[b,tmp_node_action] *\
                                 vnf_probs[0,tmp_vnf_action] + 1e-8).unsqueeze(0)

        action_logprob = tmp_logprob if action_logprob is None else\
                            torch.cat((action_logprob, tmp_logprob), 0)
        node_action[b] = tmp_node_action
        vnf_action[b] = tmp_vnf_action

    return action_logprob, node_action, vnf_action

def ComputeRewards(TDset, req_step, delay_coeff, reward_mode, B):
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

    rewards = np.zeros(B)

    for b in range(B):
        TD = TDset[b]
        if req_step >= len(TD.reqs.keys()):
            continue
        req_idx = list(TD.reqs.keys())[req_step]

        rewards[b] = TD.ComputeReward(req_idx, delay_coeff, reward_mode)

    return rewards

