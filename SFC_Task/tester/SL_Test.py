import time

import torch
import torch.nn as nn
from torch.autograd import Variable

from data.BatchGenerator import MakeEncodingBatch, MakeDecodingBatch, MakeLabelBatch
from utils.TopologyUpdater import SetTopology, UpdateGenerations, UpdateCapacities, MakeTDset
from evaluations.metrics import ComputeFails, ComputeDelayRatios
from models.model import Call_Model

def SL_test(TDset, B, model, data_spec, predict_mode, device, max_gen):
    '''
    B is the batch_size
    N is the number of nodes in the topology

    - INPUT
    TDset        : <B> class TD   , TopologyDrivers which is already set 
                                  with requests, deployments, and labels
    model        : class model
    data_spec    : { 'max_reqs'         : MR,
                     'max_depls'        : MD,
                     'max_labels'       : ML,
                     'n_req_features'   : FR,
                     'n_depl_features'  : FD,
                     'n_label_features' : FL }
    predict_mode : str            , 'NodeLevel' or 'VNFLevel'
    deivce       : str            , device that the model is running on
    max_gen      : int            , maximum limit of predictions    

    '''

    model.eval()
    softmax = nn.Softmax(dim=1)

    N = TDset[0].n_nodes

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
        hidden = None
        for gen_step in range(max_gen):
            from_node, vnf_now, vnf_all, dec_mask, testing_flag\
                                     = MakeDecodingBatch(TDset, r_step, predict_mode, B)
            if testing_flag == False:
                break

            mask = enc_mask.reshape(B,1)*dec_mask

            logit_mask = mask.reshape(B*N)

            # Run decoder
            node_logits, vnf_logits, hidden = model(enc_out, from_node,\
                                                     vnf_now, vnf_all, logit_mask, hidden)

            node_probs = softmax(node_logits)

            node_pred = torch.argmax(node_probs, dim=1)

            if predict_mode == 'NodeLevel':
                vnf_probs = None
                for b in range(B):
                    tmp_vnf_logit = vnf_logits[b, node_pred[b], :].unsqueeze(0)
                    vnf_probs = tmp_vnf_logit if vnf_probs is None else\
                                torch.cat((vnf_probs, tmp_vnf_logit),0) # <B, 2>
                vnf_probs = softmax(vnf_probs)
                vnf_pred = torch.argmax(vnf_probs, dim=1).cpu().numpy()
            else:
                vnf_pred = None

            # Update Generations
            UpdateGenerations(TDset, r_step, node_pred.cpu().numpy(),\
                                             B, vnf_pred) # Update with generation
        

        # Update Capacities
        UpdateCapacities(TDset, r_step, B)

        tmp_n_reqs, n_fails = ComputeFails(TDset, r_step, B)
        _, delay_ratios = ComputeDelayRatios(TDset, r_step, B)

        n_reqs += tmp_n_reqs
        total_n_fail += n_fails
        total_delayratio += delay_ratios

    return total_n_fail, total_delayratio, n_reqs

def SL_test_main(args, pt_model, testset):
    print("-----Testing Start-----")

    TDset = MakeTDset(args.batch_size, args.environment, args.predict_mode, args.adj_temperature,\
                        args.recurrent_delay, args.topo_path, args.sfctypes_path,\
                        args.middlebox_path)

    model = Call_Model(args, TDset[0].n_vnfs)
    model.to(args.device)

    model.Load_PTmodel(pt_model)

    start_time = time.time()
    test_fail = 0
    test_delay = 0
    n_reqs = 0

    testset_spec = testset.dataset.data_spec

    for i, (requests, deployments, labels) in enumerate(testset):
        current_batch_size = SetTopology(TDset, requests, deployments, labels, testset_spec,\
                     learning_mode='SL')

        tmp_fail, tmp_delay, tmp_reqs = SL_test(TDset, current_batch_size,\
                                 model, testset_spec, args.predict_mode, args.device, args.max_gen)

        test_fail += tmp_fail
        test_delay += tmp_delay
        n_reqs += tmp_reqs

    test_fail = float(test_fail / n_reqs)
    test_delay = float(test_delay / n_reqs)

    return test_fail, test_delay

