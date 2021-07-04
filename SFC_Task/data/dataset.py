import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pickle
from sklearn.utils import shuffle

def Split_Datasets(dataset_path, trainset_path, validset_path, testset_path, ratios):
    [train_r, valid_r, test_r] = ratios

    dataset = pd.read_pickle(dataset_path)
    dataset = shuffle(dataset)
    n_samples = len(dataset)

    n_train = int(n_samples * train_r)
    n_valid = int(n_samples * valid_r)

    trainset = dataset[0:n_train]
    validset = dataset[n_train:n_train+n_valid]
    testset = dataset[n_train+n_valid:]

    with open(trainset_path, 'wb') as new_trainset:
        pickle.dump(trainset, new_trainset)
    with open(validset_path, 'wb') as new_validset:
        pickle.dump(validset, new_validset)
    with open(testset_path, 'wb') as new_testset:
        pickle.dump(testset, new_testset)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, max_reqs, max_depls, max_labels,\
                 n_req_features, n_depl_features, n_label_features,\
                 data_path):
        self.max_reqs = max_reqs
        self.max_depls = max_depls
        self.max_labels = max_labels
        self.n_req_f = n_req_features
        self.n_depl_f = n_depl_features
        self.n_label_f = n_label_features

        self.data_spec = {}
        self.data_spec['max_reqs'] = max_reqs
        self.data_spec['max_depls'] = max_depls
        self.data_spec['max_labels'] = max_labels
        self.data_spec['n_req_features'] = n_req_features
        self.data_spec['n_depl_features'] = n_depl_features
        self.data_spec['n_label_features'] = n_label_features
        
        with open(data_path, 'rb') as data:
            self.data = pickle.load(data)

    def __getitem__(self, index):
        sample = self.data[index]
        request = sample[:self.max_reqs*self.n_req_f]
        base_idx = self.max_reqs*self.n_req_f
        deployment = sample[base_idx:base_idx+self.max_depls*self.n_depl_f]
        base_idx = base_idx + self.max_depls*self.n_depl_f
        label = sample[base_idx:base_idx+self.max_labels*self.n_label_f]
        return np.array(request), np.array(deployment), np.array(label)

    def __len__(self):
        return len(self.data)

def MyDataLoader(max_reqs, max_depls, max_labels,\
                 n_req_features, n_depl_features, n_label_features, data_path, batch_size):
    data = MyDataset(max_reqs, max_depls, max_labels,\
                    n_req_features, n_depl_features, n_label_features, data_path)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return data_loader
