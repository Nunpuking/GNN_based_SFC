import neptune
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data.BatchGenerator import MakeEncodingBatch, MakeDecodingBatch, MakeLabelBatch
from utils.TopologyUpdater import SetTopology, UpdateGenerations, UpdateCapacities,\
                                    MakeTDset
from utils.util_heo import open_log, training_manager, weights_initializer, timeSince
from utils.topology import TopologyDriver
from models.model import Call_Model
from tester.SL_Test import SL_test_main

def SL_train(TDset, B, model, optimizer, data_spec, predict_mode, device):
    '''
    B is the batch_size
    N is the number of nodes in the topology

    - INPUT
    TDset        : <B> class TD   , TopologyDrivers which is already set 
                                  with requests, deployments, and labels
    B            : int
    model        : class model
    optimizer    : class optimizer
    data_spec    : { 'max_reqs'         : MR,
                     'max_depls'        : MD,
                     'max_labels'       : ML,
                     'n_req_features'   : FR,
                     'n_depl_features'  : FD,
                     'n_label_features' : FL }
    predict_mode : str            , 'NodeLevel' or 'VNFLevel'
    deivce       : str            , device that the model is running on
    
    '''


    model.train()
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss(reduction='none')

    N = TDset[0].n_nodes

    total_loss = 0
    n_predictions = 0

    for r_step in range(data_spec['max_reqs']):
        annotation, A_out, A_in, enc_mask, training_flag = MakeEncodingBatch(TDset, r_step, B)
        if training_flag == False:
            break

        # Encoding Stage
        enc_out = model.encoder(annotation, A_out, A_in)

        # Decoding Stage
        train_loss = 0
        hidden = None
        for gen_step in range(1000):
            from_node, vnf_now, vnf_all, dec_mask, training_flag\
                                     = MakeDecodingBatch(TDset, r_step, predict_mode, B)
            if training_flag == False:
                break

            label_node, label_vnf = MakeLabelBatch(TDset, r_step, gen_step, B)
            label_node = Variable(torch.from_numpy(label_node).type(torch.long)).to(device)
            label_vnf = Variable(torch.from_numpy(label_vnf).type(torch.long)).to(device)

            mask = enc_mask.reshape(B,1)*dec_mask
            loss_mask = [1 if sum(vals) > 0 else 0 for vals in mask]

            n_predictions += sum(loss_mask)

            loss_mask = Variable(torch.tensor(loss_mask)).to(device)
            logit_mask = mask.reshape(B*N)

            # Run decoder
            node_logits, vnf_logits, hidden = model(enc_out, from_node,\
                                                     vnf_now, vnf_all, logit_mask, hidden)

            # Compute losses
            node_loss = loss_mask * criterion(node_logits, label_node)
            node_probs = softmax(node_logits)

            node_pred = torch.argmax(node_probs, dim=1)

            if predict_mode == 'NodeLevel':
                vnf_probs = None
                for b in range(B):
                    tmp_vnf_logit = vnf_logits[b, node_pred[b], :].unsqueeze(0)
                    vnf_probs = tmp_vnf_logit if vnf_probs is None else\
                                torch.cat((vnf_probs, tmp_vnf_logit),0) # <B, 2>
                vnf_loss = loss_mask * criterion(vnf_probs, label_vnf)
                vnf_probs = softmax(vnf_probs)
                vnf_pred = torch.argmax(vnf_probs, dim=1)
                vnf_loss = vnf_loss.sum()
            else:
                vnf_loss = 0

            train_loss += node_loss.sum() + vnf_loss

            # Update Generations
            UpdateGenerations(TDset, r_step, label_node.cpu().numpy(),\
                                             B, label_vnf.cpu().numpy()) # Update with label


        # Backprop and Update
        train_loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 2)
        optimizer.step()
        optimizer.zero_grad()

        # Update Capacities
        UpdateCapacities(TDset, r_step, B)

        total_loss += train_loss.item()

    return total_loss, n_predictions

def SL_train_main(args, trainset, validset, neptune_log_names, checkpoint=None):
    print("-----Training Start-----")
    open_log(args.save_dir, dir=True, message="result_dir")
    open_log(args.save_subdir, dir=True, message="subdir")
    train_log = open_log(args.train_log_path, message="train")
    valid_log = open_log(args.valid_log_path, message="valid")

    train_log.write("{}\t{}\t{}\n".format('Iters', 'Loss', 'Time'))
    valid_log.write("{}\t{}\t{}\t{}\n".format(\
                        'Iters', 'Fail', 'Delay', 'Time'))

    nt_train_loss, nt_valid_fail, nt_valid_delay = neptune_log_names

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
                continue
                
            current_batch_size = SetTopology(TDset, requests, deployments, labels, trainset_spec,\
                         learning_mode='SL')

            loss, n_pred = SL_train(TDset, current_batch_size, model, optimizer, trainset_spec,\
                                     args.predict_mode, args.device)

            print_loss += loss
            print_n_pred += n_pred
            print_iters += current_batch_size
            valid_iters += current_batch_size
            n_iters += current_batch_size

            if print_iters >= args.print_iter:
                print_iters -= args.print_iter

                avg_loss = float(print_loss/print_n_pred)
                print_loss = 0
                print_n_pred = 0

                print("ITER/EPOCH {}/{} | LOSS {:.4f} | BEST {:.4f} | PAT. {} LR_DECAY {} | {}".format(n_iters, epoch, avg_loss, best_eval, manager.n_patience, manager.n_lr_decay,\
                    timeSince(start_time)))
                neptune.log_metric(nt_train_loss, avg_loss)

                train_log.write("{}\t{}\t{}\n".format(\
                    n_iters, avg_loss, timeSince(start_time)))

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

                valid_fail, valid_delay = SL_test_main(args, model, validset)

                print("FAIL     : {:.4f}".format(valid_fail))
                print("DELAY    : {:.4f}".format(valid_delay))
                print("=========================================")
                neptune.log_metric(nt_valid_fail, valid_fail)
                neptune.log_metric(nt_valid_delay, valid_delay)
                valid_log.write("{}\t{}\t{}\t{}\n".format(\
                    n_iters, valid_fail, valid_delay, timeSince(start_time)))

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

