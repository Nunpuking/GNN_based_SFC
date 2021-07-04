import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from data.BatchGenerator import MakeEncodingBatch, MakeDecodingBatch, MakeLabelBatch
from utils.TopologyUpdater import SetTopology, UpdateGenerations, UpdateCapacities,\
                                    MakeTDset, RandomTopology, RandomDeployment
from evaluations.metrics import ComputeFails, ComputeDelayRatios
from trainer.RL_utils import ActionSample, ComputeRewards
from models.model import Call_Model

def REINFORCE_test(TDset, B, model, data_spec, predict_mode, device, max_gen, delay_coeff):
    '''
    B is the batch_size
    N is the number of nodes in the topology

    - INPUT
    TDset           : <B> class TD, TopologyDrivers which is already set 
                               with requests, deployments, and labels
    model           : class model
    data_spec       : { 'max_reqs'         : MR,
                        'max_depls'        : MD,
                        'max_labels'       : ML,
                        'n_req_features'   : FR,
                        'n_depl_features'  : FD,
                        'n_label_features' : FL }
    predict_mode    : str          , 'NodeLevel' or 'VNFLevel'
    device          : str          , device that the model is running on
    max_gen         : int          , maximum generation step
    delay_coeff     : float        , the controlling parameter of delay reward

    '''

    model.eval()
    softmax = nn.Softmax(dim=1)

    N = TDset[0].n_nodes

    total_reward = 0
    total_n_fail = 0
    total_delayratio = 0
    n_reqs = 0
    for r_step in range(data_spec['max_reqs']):
        annotation, A_out, A_in, enc_mask, testing_flag = MakeEncodingBatch(TDset, r_step, B)
        if testing_flag == False:
            break

        # Encoding Stage
        enc_out = model.encoder(annotation, A_out, A_in)

        # Decoding Stage
        rewards = None
        saved_log_probs = None
        hidden = None
        for gen_step in range(max_gen):
            from_node, vnf_now, vnf_all, dec_mask, testing_flag\
                                        = MakeDecodingBatch(TDset, r_step, predict_mode, B)
            if testing_flag == False:
                break

            mask = enc_mask.reshape(B,1)*dec_mask
            sample_mask = [1 if sum(vals) > 0 else 0 for vals in mask]
            sample_mask_tensor = Variable(torch.tensor(sample_mask)).to(device)

            logit_mask = mask.reshape(B*N)

            # Run decoder
            node_logits, vnf_logits, hidden = model(enc_out, from_node,\
                                                    vnf_now, vnf_all, logit_mask, hidden)

            # Choose actions
            action_logprob, node_action, vnf_action =\
                                ActionSample(node_logits, vnf_logits, mask, epsilon=0, B=B)

            # Update Generations
            if predict_mode == 'NodeLevel':
                UpdateGenerations(TDset, r_step, node_action, B, vnf_action)
            else:
                UpdateGenerations(TDset, r_step, node_action, B, None)

            # Save rewards
            action_reward = ComputeRewards(TDset, r_step, delay_coeff, reward_mode='REINFORCE', B=B)
            action_reward = (sample_mask*action_reward).reshape(B,1)

            rewards = action_reward if rewards is None else\
                                np.concatenate((rewards, action_reward), 1)

        # Update Capacities
        UpdateCapacities(TDset, r_step, B)

        total_reward += np.sum(rewards) if rewards is not None else 0 # 0 is for fail case

        tmp_n_reqs, n_fails = ComputeFails(TDset, r_step, B)
        _, delay_ratios = ComputeDelayRatios(TDset, r_step, B)

        n_reqs += tmp_n_reqs
        total_n_fail += n_fails
        total_delayratio += delay_ratios

    return total_reward, total_n_fail, total_delayratio, n_reqs

def REINFORCE_test_main(args, pt_model, testset):
    print("-----Testing Start-----")

    TDset = MakeTDset(args.batch_size, args.environment, args.predict_mode, args.adj_temperature,\
                        args.recurrent_delay, args.topo_path, args.sfctypes_path,\
                        args.middlebox_path)

    model = Call_Model(args, TDset[0].n_vnfs)
    model.to(args.device)

    model.Load_PTmodel(pt_model)

    start_time = time.time()
    test_reward = 0
    test_fail = 0
    test_delay = 0
    n_reqs = 0

    testset_spec = testset.dataset.data_spec

    early_stop = False
    for i, (requests, deployments, labels) in enumerate(testset):
        if args.topology_change_mode_test == 1:
            RandomTopology(TDset, args.random_topology_test_dir, args.sfctypes_path,\
                         args.middlebox_path)

        current_batch_size = SetTopology(TDset, requests, deployments, labels, testset_spec,\
                     learning_mode='RL')

        if args.deployment_change_mode_test == 1:
            RandomDeployment(TDset)

        tmp_reward, tmp_fail, tmp_delay, tmp_reqs =\
                     REINFORCE_test(TDset, current_batch_size, model, testset_spec,\
                                 args.predict_mode, args.device, args.max_gen, args.delay_coeff)

        test_reward += tmp_reward
        test_fail += tmp_fail
        test_delay += tmp_delay
        n_reqs += tmp_reqs

    test_reward = float(test_reward / n_reqs)
    test_fail = float(test_fail / n_reqs)
    test_delay = float(test_delay / n_reqs)

    return test_reward, test_fail, test_delay


