import argparse
import neptune
import os
import torch
import numpy as np

from trainer.SL_Train import SL_train_main
from tester.SL_Test import SL_test_main
from trainer.REINFORCE_Train import REINFORCE_train_main
from tester.REINFORCE_Test import REINFORCE_test_main
from utils.util_heo import CountDown, path_settings
from utils.topology import TopologyDriver
from utils.TopologyUpdater import MakeTDset
from data.generate_dataset import PreProcessing
from data.dataset import Split_Datasets, MyDataLoader

def DataPreProcessing(args):
    print("Data Preprocessing ..")
    CountDown(5)
    
    TD = MakeTDset(args.batch_size, args.envrionment, args.predict_mode, args.temperature,\
                        args.recurrent_delay, args.topo_path, args.sfctypes_path,\
                        args.middlebox_path)[0]

    PreProcessing(TD.vnf_spec, args.data_req_features, args.data_depl_features,\
                args.data_label_features, args.raw_request_path, args.raw_deployment_path,\
                args.raw_label_path, dataset_path, args.data_dir, load_idxes=True)

def train(args):
    print("=====TRAINING START=====")
    print("n_running : ", args.n_running)
   
    for n_run in range(args.n_running):
        args.model_path, args.train_log_path, args.valid_log_path = path_settings(args.save_subdir,\
                args.running_mode, n_run)

        checkpoint = None
        if args.load_checkpoint == 1:
            checkpoint = torch.load(args.model_path)

        if args.load_pt_model != '' :
            print("{} pre-trained model will loaded, are you sure??".format(args.load_pt_model))
            CountDown(5)
            checkpoint = torch.load(args.load_pt_model)
            checkpoint['epoch'] = 0
            checkpoint['iters'] = 0
        
        if args.learning_fashion in ['SL']:
            nt_train_loss = 'TRAIN' + str(n_run) + '_LOSS'
            nt_valid_fail = 'VALID' + str(n_run) + '_FAIL'
            nt_valid_delay = 'VALID' + str(n_run) + '_DELAY'
            neptune_log_names = (nt_train_loss, nt_valid_fail, nt_valid_delay)
            _,_ = SL_train_main(args, trainset, validset, neptune_log_names, checkpoint)

        elif args.learning_fashion in ['REINFORCE']:
            nt_train_loss = 'TRAIN' + str(n_run) + '_LOSS'
            nt_train_reward = 'TRAIN' + str(n_run) + '_REWARD'
            nt_valid_reward = 'VALID' + str(n_run) + '_REWARD'
            nt_valid_fail = 'VALID' + str(n_run) + '_FAIL'
            nt_valid_delay = 'VALID' + str(n_run) + '_DELAY'
            neptune_log_names = (nt_train_loss, nt_train_reward, nt_valid_reward,\
                                nt_valid_fail, nt_valid_delay)
            _,_ = REINFORCE_train_main(args, trainset, validset, neptune_log_names, checkpoint)

def test(args):
    print("=====TESTING START=====")
    print("n_running : ", args.n_running)
    final_fail = np.zeros(args.n_running)
    final_delay = np.zeros(args.n_running)

    for n_run in range(args.n_running):
        args.model_path = path_settings(args.load_subdir, args.running_mode, n_run)

        load_model_path = args.model_path + '.best.pth'
        checkpoint = torch.load(load_model_path)
        loaded_model = checkpoint['model']
        loaded_model.to(args.device)
        print("MODEL : ", load_model_path)

        if args.learning_fashion in ['SL']:
            fail, delay = SL_test_main(args, loaded_model, testset)
        elif args.learning_fashion in ['REINFORCE']:
            _, fail, delay = REINFORCE_test_main(args, loaded_model, testset)

        final_fail[n_run] = fail
        final_delay[n_run] = delay

    mean_fail = np.mean(final_fail)
    std_fail = np.std(final_fail)
    mean_delay = np.mean(final_delay)
    std_delay = np.std(final_delay)

    print("MODEL : ", load_model_path)
    print("Std. FAIL {} | Std. DELAY {}".format(std_fail, std_delay))
    print("Mean FAIL {} | Mean DELAY {}".format(mean_fail, mean_delay))


parser = argparse.ArgumentParser()
parser.add_argument("--running_mode", type=str, default='train') #
parser.add_argument("--task_name", type=str, default='SFC') #
parser.add_argument("--model_name", type=str, default='') #
parser.add_argument("--predict_mode", type=str, default='') #
parser.add_argument("--learning_fashion", type=str, default='') #
parser.add_argument("--environment", type=str, default='Simulation') #

parser.add_argument("--topology_change_mode", type=int, default=0) #
parser.add_argument("--deployment_change_mode", type=int, default=0) #
parser.add_argument("--topology_change_mode_test", type=int, default=0) #
parser.add_argument("--deployment_change_mode_test", type=int, default=0) #

# Load & Save or Dataset paths
parser.add_argument("--topo_path", type=str, default='./data/datasets/inet2') #
parser.add_argument("--sfctypes_path", type=str, default='./data/datasets/sfctypes') #
parser.add_argument("--middlebox_path", type=str, default='./data/datasets/middlebox-spec') #

parser.add_argument("--data_dir", type=str, default='./data/datasets/') #
parser.add_argument("--dataset_file", type=str, default='processed_dataset.pickle') #
parser.add_argument("--trainset_file", type=str, default='20190530_trainset.pkl') #
parser.add_argument("--validset_file", type=str, default='20190530_validset.pkl') #
parser.add_argument("--testset_file", type=str, default='20190530_testset.pkl') #
parser.add_argument("--data_split", type=int, default=0) #
parser.add_argument("--save_dir", type=str, default='./results/') #
parser.add_argument("--load_dir", type=str, default='./backups/') #
parser.add_argument("--random_topology_dir", type=str,\
            default='./data/datasets/RandomTopologies/') #
parser.add_argument("--random_topology_test_dir", type=str,\
            default='./data/datasets/RandomTopologies/for_test/') #

parser.add_argument("--raw_request_path", type=str, default='./data/datasets/20190530-requests.csv')
parser.add_argument("--raw_deployment_path", type=str,\
            default='./data/datasets/20190530-nodeinfo.csv')
parser.add_argument("--raw_label_path", type=str, default='./data/datasets/20190530-routeinfo.csv')

parser.add_argument("--load_checkpoint", type=int, default=0) #
parser.add_argument("--load_pt_model", type=str, default='') #

# Training Configurations
parser.add_argument("--n_running", type=int, default=0) #
parser.add_argument("--epochs", type=int, default=100) #
parser.add_argument("--print_iter", type=int, default=50) #
parser.add_argument("--valid_iter", type=int, default=250) #
parser.add_argument("--batch_size", type=int, default=20) #

parser.add_argument("--patience", type=int, default=10) #
parser.add_argument("--scheduled_lr_decay", type=int, default=0) #
parser.add_argument("--lr_decay", type=int, default=2) #
parser.add_argument("--lr_decay_ratio", type=float, default=0.1) #
parser.add_argument("--lr", type=float, default=1e-3) #
parser.add_argument("--opt", type=str, default='Adam') #

parser.add_argument("--max_gen", type=int, default=30) #

parser.add_argument("--rl_epsilon", type=float, default=0.01) #
parser.add_argument("--delay_coeff", type=float, default=0) #
parser.add_argument("--discount_factor", type=float, default=0.999) #

# Model Configurations
parser.add_argument("--GRU_steps", type=int, default=5) #
parser.add_argument("--node_state_dim", type=int, default=128) #
parser.add_argument("--posenc_node_dim", type=int, default=4) #
parser.add_argument("--max_n_nodes", type=int, default=50) #
parser.add_argument("--recurrent_delay", type=float, default=0.1) #
parser.add_argument("--adj_temperature", type=float, default=2) #
parser.add_argument("--vnf_dim", type=int, default=5) #

# Dataset Spec
parser.add_argument("--data_max_reqs", type=int, default=41) #
parser.add_argument("--data_max_depls", type=int, default=20) #
parser.add_argument("--data_max_labels", type=int, default=116) # 
parser.add_argument("--data_req_features", type=int, default=7) #
parser.add_argument("--data_depl_features", type=int, default=3) #
parser.add_argument("--data_label_features", type=int, default=3) #

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.device = torch.device("cuda" if args.cuda else "cpu")

args.subdir = args.learning_fashion + '_' + args.task_name + '_' + args.predict_mode + '_'\
                + args.opt + '_' + str(args.lr) + 'LR_' + str(args.batch_size) + 'Batch'
if args.learning_fashion in ['REINFORCE']:
    args.subdir += '_' + str(args.topology_change_mode) + 'TCM_'\
                     + str(args.deployment_change_mode) + 'DCM_'\
                    + str(args.delay_coeff) + 'DelayCoeff'
    if args.load_pt_model != '':
        args.subdir += '_PT'

args.save_subdir = args.save_dir + args.subdir
args.load_subdir = args.load_dir + args.subdir

dataset_path = args.data_dir + args.dataset_file
trainset_path = args.data_dir + args.trainset_file
validset_path = args.data_dir + args.validset_file
testset_path = args.data_dir + args.testset_file

if args.data_split == 1:
    print("WARNING : Datasets will be splitting and overwritten")
    CountDown(5)

    ratios = [0.87, 0.03, 0.1]
    Split_Datasets(dataset_path, trainset_path, validset_path, testset_path, ratios)

trainset = MyDataLoader(args.data_max_reqs, args.data_max_depls, args.data_max_labels,\
                        args.data_req_features, args.data_depl_features, args.data_label_features,\
                        trainset_path, args.batch_size)
validset = MyDataLoader(args.data_max_reqs, args.data_max_depls, args.data_max_labels,\
                        args.data_req_features, args.data_depl_features, args.data_label_features,\
                        validset_path, args.batch_size)
testset = MyDataLoader(args.data_max_reqs, args.data_max_depls, args.data_max_labels,\
                        args.data_req_features, args.data_depl_features, args.data_label_features,\
                        testset_path, args.batch_size)

if args.running_mode == 'train':
    neptune.init('nunpuking/IntegratedSFC')
    neptune.create_experiment(args.subdir)

    train(args)
    args.running_mode = 'test'
    test(args)

elif args.running_mode == 'test':
    test(args)

elif args.running_mode == 'data_preprocessing':
    DataPreProcessing(args)

else:
    raise SyntaxError("ERROR: Worng running mode name {}".format(args.running_mode))
