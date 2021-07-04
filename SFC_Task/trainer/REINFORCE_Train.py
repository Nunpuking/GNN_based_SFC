import neptune
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from trainer.RL_utils import ActionSample, ComputeRewards
from data.BatchGenerator import MakeEncodingBatch, MakeDecodingBatch, MakeLabelBatch,\
                                CheckPossibility
from utils.TopologyUpdater import SetTopology, UpdateGenerations, UpdateCapacities,\
                                     MakeTDset, RandomTopology, RandomDeployment
from utils.util_heo import open_log, training_manager, weights_initializer, timeSince
from models.model import Call_Model
from tester.REINFORCE_Test import REINFORCE_test_main

def REINFORCE_train(TDset, B, model, optimizer, data_spec, predict_mode, device, max_gen,\
                     epsilon, delay_coeff, discount_factor):
    '''
    B is the batch_size
    N is the number of nodes in the topology

    - INPUT
    TDset           : <B> class TD, TopologyDrivers which is already set 
                               with requests, deployments, and labels
    B               : int
    model           : class model
    optimizer       : class optimizer
    data_spec       : { 'max_reqs'         : MR,
                        'max_depls'        : MD,
                        'max_labels'       : ML,
                        'n_req_features'   : FR,
                        'n_depl_features'  : FD,
                        'n_label_features' : FL }
    predict_mode    : str          , 'NodeLevel' or 'VNFLevel'
    device          : str          , device that the model is running on
    max_gen         : int          , maximum generation step
    epsilon         : float        , probability for exploration
    delay_coeff     : float        , the controlling parameter of delay reward
    discount_factor : float        , discount factor for computaton of returns

    '''

    model.train()
    softmax = nn.Softmax(dim=1)

    N = TDset[0].n_nodes

    total_reward = 0
    total_loss = 0
    n_reqs = 0
    n_pred = 0
    for r_step in range(data_spec['max_reqs']):
        annotation, A_out, A_in, enc_mask, training_flag = MakeEncodingBatch(TDset, r_step, B)

        if training_flag == False:
            break

        n_reqs += np.sum(enc_mask)

        # Encoding Stage
        enc_out = model.encoder(annotation, A_out, A_in)

        # Decoding Stage
        rewards = None
        saved_log_probs = None
        hidden = None
        for gen_step in range(max_gen):
            from_node, vnf_now, vnf_all, dec_mask, training_flag\
                                        = MakeDecodingBatch(TDset, r_step, predict_mode, B)
            if training_flag == False:
                break

            mask = enc_mask.reshape(B,1)*dec_mask
            sample_mask = [1 if sum(vals) > 0 else 0 for vals in mask]
            sample_mask_tensor = Variable(torch.tensor(sample_mask).type(torch.float)).to(device)

            n_pred += sum(sample_mask)
            
            logit_mask = mask.reshape(B*N)

            # Run decoder
            node_logits, vnf_logits, hidden = model(enc_out, from_node,\
                                                    vnf_now, vnf_all, logit_mask, hidden)

            # Choose actions
            action_logprob, node_action, vnf_action =\
                                ActionSample(node_logits, vnf_logits, mask, epsilon, B)

            # Update Generations
            if predict_mode == 'NodeLevel':
                UpdateGenerations(TDset, r_step, node_action, B, vnf_action)
            else:
                UpdateGenerations(TDset, r_step, node_action, B, None)

            # Save transition (logprob, reward)
            action_logprob = (sample_mask_tensor*action_logprob).unsqueeze(1)

            saved_log_probs = action_logprob if saved_log_probs is None else\
                                torch.cat((saved_log_probs, action_logprob), 1)

            action_reward = ComputeRewards(TDset, r_step, delay_coeff, reward_mode='REINFORCE', B=B)
            action_reward = (sample_mask*action_reward).reshape(B,1)

            rewards = action_reward if rewards is None else\
                                np.concatenate((rewards, action_reward), 1)

        if rewards is None:
            continue

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
            logprob = saved_log_probs[:,i].type(torch.double)
            R = returns[:,i]
            train_loss += (-logprob*R).sum()

        train_loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 2)
        optimizer.step()
        optimizer.zero_grad()

        # Update Capacities
        UpdateCapacities(TDset, r_step, B)

        total_loss += train_loss.item()
        total_reward += np.sum(rewards)
        
    return total_loss, n_pred, total_reward, n_reqs

def REINFORCE_train_main(args, trainset, validset, neptune_log_names, checkpoint=None):
    print("-----Training Start-----")
    open_log(args.save_dir, dir=True, message="result_dir")
    open_log(args.save_subdir, dir=True, message="subdir")
    train_log = open_log(args.train_log_path, message="train")
    valid_log = open_log(args.valid_log_path, message="valid")

    train_log.write("{}\t{}\t{}\t{}\n".format('Iters', 'Loss', 'Reward', 'Time'))
    valid_log.write("{}\t{}\t{}\t{}\t{}\n".format(\
                        'Iters', 'Reward', 'Fail', 'Delay', 'Time'))

    nt_train_loss, nt_train_reward, nt_valid_reward,\
                     nt_valid_fail, nt_valid_delay = neptune_log_names

    # No LR decay, Early Stopping
    args.patience = 100000
    manager = training_manager(args)

    TDset = MakeTDset(args.batch_size, args.environment, args.predict_mode, args.adj_temperature,\
                        args.recurrent_delay, args.topo_path, args.sfctypes_path,\
                        args.middlebox_path)

    checkpoint_epoch = 0
    skip_iters = False
    if checkpoint is not None:
        print("Model and optimizer are loaded from checkpoint!")
        checkpoint_epoch = checkpoint['epoch']
        checkpoint_iter = checkpoint['iters']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        skip_iters = True

    else:
        model = Call_Model(args, TDset[0].n_vnfs)
        model.apply(weights_initializer)

        print( sum( p.numel() for p in model.parameters() if p.requires_grad ) )

        if args.opt == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.opt == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        elif args.opt == 'Adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        elif args.opt == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    model.to(args.device)

    start_time = time.time()
    print_loss = 0
    print_n_pred = 0
    print_reward = 0
    print_n_reqs = 0
    best_eval = 10000

    n_iters = 0
    print_iters = 0
    valid_iters = 0

    trainset_spec = trainset.dataset.data_spec

    early_stop = False
    for epoch in range(checkpoint_epoch, args.epochs):
        if early_stop is True:
            break
        model.train()

        for n_iter, (requests, deployments, labels) in enumerate(trainset):
            if skip_iters == True and n_iter < checkpoint_iter:
                n_iters += args.batch_size
                continue

            if args.topology_change_mode == 1:
                RandomTopology(TDset, args.random_topology_dir, args.sfctypes_path,\
                                 args.middlebox_path)

            current_batch_size = SetTopology(TDset, requests, deployments, labels, trainset_spec,\
                         learning_mode='RL')

            if args.deployment_change_mode == 1:
                RandomDeployment(TDset, current_batch_size)

            loss, n_pred, reward, n_reqs = REINFORCE_train(TDset, current_batch_size, model,\
                                     optimizer, trainset_spec,\
                                     args.predict_mode, args.device, args.max_gen, args.rl_epsilon,\
                                     args.delay_coeff, args.discount_factor)

            print_loss += loss
            print_n_pred += n_pred
            print_reward += reward
            print_n_reqs += n_reqs

            print_iters += current_batch_size
            valid_iters += current_batch_size
            n_iters += current_batch_size

            if print_iters >= args.print_iter:
                print_iters -= args.print_iter
        
                avg_loss = float(print_loss/print_n_pred)
                avg_reward = float(print_reward/print_n_reqs)
                print_loss = 0
                print_n_pred = 0
                print_reward = 0
                print_n_reqs = 0

                print("ITER/EPOCH {}/{} | LOSS {:.4f} REWARD {:.4f} | BEST {:.4f} | PAT. {} LR_DECAY {} | {}".format(n_iters, epoch, avg_loss, avg_reward, best_eval, manager.n_patience,\
                 manager.n_lr_decay, timeSince(start_time)))
                neptune.log_metric(nt_train_loss, avg_loss)
                neptune.log_metric(nt_train_reward, avg_reward)

                train_log.write("{}\t{}\t{}\t{}\n".format(\
                    n_iter, avg_loss, avg_reward, timeSince(start_time)))

            if valid_iters >= args.valid_iter:
                valid_iters -= args.valid_iter

                torch.save({'epoch':epoch,
                            'iters':n_iter,
                            'model':model,
                            'optimizer':optimizer,
                            }, args.model_path)
                print("=========================================")
                print("=============Validation Starts===========")
                print(args.save_subdir)

                valid_reward, valid_fail, valid_delay =\
                                         REINFORCE_test_main(args, model, validset)

                print("REWARD   : {:.4f}".format(valid_reward))
                print("FAIL     : {:.4f}".format(valid_fail))
                print("DELAY    : {:.4f}".format(valid_delay))
                print("=========================================")
                neptune.log_metric(nt_valid_reward, valid_reward)
                neptune.log_metric(nt_valid_fail, valid_fail)
                neptune.log_metric(nt_valid_delay, valid_delay)
                valid_log.write("{}\t{}\t{}\t{}\t{}\n".format(\
                    n_iter, valid_reward, valid_fail, valid_delay, timeSince(start_time)))

                valid_eval = 10*valid_fail + valid_delay

                if best_eval > valid_eval:
                    print("We find the new best model")
                    best_eval = valid_eval
                    torch.save({'epoch':epoch,
                                'iters':n_iter,
                                'model':model,
                                'optimizer':optimizer,
                                }, args.model_path + '.best.pth')
                    manager.n_patience = 0
                else:
                    early_stop, optimizer = manager.patience_step(optimizer)
            skip_iters = False
    return train_log, valid_log

