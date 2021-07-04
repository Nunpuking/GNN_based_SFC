import os
import numpy as np
import pandas as pd
import time
import copy
import pickle

from utils.topology import TopologyDriver
from utils.util_heo import timeSince
from data.dataset import MyDataLoader

def TransformRawRequest(raw_req_inst, req_idx):
    [arrivaltime, _, src, dst, traffic, maxlat, _, _, _, _, _, sfcid]\
             = raw_req_inst

    arrivaltime = int(arrivaltime)
    src = int(src)
    dst = int(dst)
    traffic = int(traffic)
    maxlat = int(maxlat)
    sfcid = int(sfcid)

    return [req_idx, arrivaltime, src, dst, traffic, maxlat, sfcid]

def TransformRawDeployment(raw_depl_inst, vnf_spec):
    [arrivaltime, node_id, vnf_type, n_inst] = raw_depl_inst
    
    node_id = int(node_id) - 1
    type_num = vnf_spec[vnf_type]['type_num']
    n_inst = int(n_inst)

    return [node_id, type_num, n_inst]

def TransformRawLabel(raw_label_inst, vnf_spec):
    [arrivaltime, req_idx, vnf_type, node_id] = raw_label_inst

    req_idx = int(req_idx) - 1
    type_num = vnf_spec[vnf_type]['type_num']
    node_id = int(node_id) - 1

    return [req_idx, type_num, node_id]

def Indexing(dataset, mode, except_cases=None):
    idxes = {}
    tmp_idxes = []
    history = []
    except_cases = [dataset[0][0]] if mode == 'request' else except_cases
    last_arrivaltime = 0
    for line_idx, inst in enumerate(dataset):
        [arrivaltime, duration, *_] = inst

        tmp_idxes.append(line_idx)
        
        endtime = arrivaltime + duration if mode == 'request' else arrivaltime
            
        history.append((endtime, arrivaltime, line_idx))
            
        copy_history = copy.deepcopy(history)
        for (tmp_endtime, tmp_arrivaltime, tmp_line_idx) in history:
            if tmp_endtime < arrivaltime:
                copy_history.remove((tmp_endtime, tmp_arrivaltime, tmp_line_idx))
                tmp_idxes.remove(tmp_line_idx)

        history = copy.deepcopy(copy_history)
        if mode == 'request' and arrivaltime == last_arrivaltime:
            except_cases.append(arrivaltime)

        idxes[arrivaltime] = copy.deepcopy(tmp_idxes)
        last_arrivaltime = arrivaltime

    for remove_arrivaltime in except_cases:
        if remove_arrivaltime in idxes.keys():
            idxes.pop(remove_arrivaltime)

    return idxes, except_cases

def IndexesIO(path, mode, idxes=None):
    if mode == 'save':
        with open(path, 'wb') as idx_path:
            pickle.dump(idxes, idx_path)
    else:
        with open(path, 'rb') as idx_path:
            idxes = pickle.load(idx_path)
        return idxes

def MaxIndex(idxes):
    max_insts = 0
    for arrivaltime, insts in idxes.items():
        if len(insts) > max_insts:
            max_insts = len(insts)
    return max_insts

def PreProcessing(vnf_spec, n_req_features, n_depl_features, n_label_features,\
                 request_path, deployment_path, label_path, processed_dataset_path,\
                 data_dir, load_idxes=False):
    raw_req_set = np.array(pd.read_csv(request_path, skiprows=0))
    raw_depl_set = np.array(pd.read_csv(deployment_path, skiprows=0))
    raw_label_set = np.array(pd.read_csv(label_path, skiprows=0))

    if load_idxes == False:
        print("Indexing Requests...")
        req_idxes, except_cases = Indexing(raw_req_set, mode='request')
        print("Indexing Deployments...")
        depl_idxes, _ = Indexing(raw_depl_set, mode='deployment', except_cases=except_cases)
        print("Indexing Labels...")
        label_idxes, _ = Indexing(raw_label_set, mode='label', except_cases=except_cases)

        # Save index sets
        print("Saving index sets...")
        IndexesIO(data_dir+'req_idxes.pickle', mode='save', idxes=req_idxes)
        IndexesIO(data_dir+'depl_idxes.pickle', mode='save', idxes=depl_idxes)
        IndexesIO(data_dir+'label_idxes.pickle', mode='save', idxes=label_idxes)
    else:
        if not os.path.exists(data_dir+'req_idxes.pickle'):
            raise SyntaxError("Preprocessed Index files are not existed..! Index them firstly!")
        req_idxes = IndexesIO(data_dir+'req_idxes.pickle', mode='load')
        depl_idxes = IndexesIO(data_dir+'depl_idxes.pickle', mode='load')
        label_idxes = IndexesIO(data_dir+'label_idxes.pickle', mode='load')

    n_data = len(req_idxes.keys())
    max_reqs = MaxIndex(req_idxes)
    max_depls = MaxIndex(depl_idxes)
    max_labels = MaxIndex(label_idxes)

    processed_dataset = [0]*n_data
    start_time = time.time()
    for n_rawline, arrivaltime in enumerate(req_idxes.keys()):
        tmp_req_idxes = req_idxes[arrivaltime]
        tmp_depl_idxes = depl_idxes[arrivaltime]
        tmp_label_idxes = label_idxes[arrivaltime]

        tmp_dataline = [0]*(n_req_features*max_reqs + n_depl_features*max_depls\
                                 + n_label_features*max_labels)

        for i, req_idx in enumerate(tmp_req_idxes):
            req_inst = TransformRawRequest(raw_req_set[req_idx], req_idx)
            tmp_dataline[i*n_req_features:(i+1)*n_req_features] = req_inst

        base_idx = max_reqs*n_req_features
        for i, depl_idx in enumerate(tmp_depl_idxes):
            depl_inst = TransformRawDeployment(raw_depl_set[depl_idx], vnf_spec)
            tmp_dataline[base_idx+i*n_depl_features:base_idx+(i+1)*n_depl_features] = depl_inst

        base_idx = base_idx + max_depls*n_depl_features
        for i, label_idx in enumerate(tmp_label_idxes):
            label_inst = TransformRawLabel(raw_label_set[label_idx], vnf_spec)
            tmp_dataline[base_idx+i*n_label_features:base_idx+(i+1)*n_label_features] = label_inst

        processed_dataset[n_rawline] = tmp_dataline

        if n_rawline % 100 == 0:
            print("{}/{} are processed".format(n_rawline, n_data))

    print("Saving Final Dataset..")
    with open(processed_dataset_path, 'wb') as new_dataset:
        pickle.dump(processed_dataset, new_dataset)
    
    print("Done!")

if __name__=='__main__':

    task_mode = 'NodeLevel'

    topo_path = '../data/inet2'
    middlebox_path = '../data/middlebox-spec2'
    sfctypes_path = '../data/sfctypes'

    TD = TopologyDriver(environment='Simulation', task_mode=task_mode,\
                        temperature=0.1, recurrent_delay=0.1)
    TD.InitializeTopology(topo_path, middlebox_path, sfctypes_path)

    max_reqs = 41
    max_depls = 20
    max_labels = 116

    n_req_features = 7 # req_idx, arrivaltime, src, dst, traffic, maxlat, sfcid
    n_depl_features = 3 # node_id, vnf_type, n_insts
    n_label_features = 3 # req_idx, vnf_type, node_id
    
    request_path = '../data/20190530-requests.csv'
    deployment_path = '../data/20190530-nodeinfo.csv'
    label_path = '../data/20190530-routeinfo.csv'

    processed_dataset_path = './data/processed_dataset.pickle'
    batch_size = 3
    temperature = 2
    recurrent_delay = 0.1
    #PreProcessing(TD.vnf_spec, n_req_features, n_depl_features, n_label_features,\
    #             request_path, deployment_path, label_path, processed_dataset_path, load_idxes=True)
    
    dataset = MyDataLoader(max_reqs, max_depls, max_labels, n_req_features, n_depl_features,\
                        n_label_features, processed_dataset_path, batch_size)

    TDset = [TD]*batch_size

    for i, (request, deployment, label) in enumerate(dataset):
        request = request[0].tolist()
        deployment = deployment[0].tolist()
        label = label[0].tolist()

        TD.ClearRequests()
        TD.ClearDeployments()

        for j in range(max_reqs):
            tmp_request = request[j*n_req_features:(j+1)*n_req_features]
            if sum(tmp_request) != 0:
                [req_idx, arrivaltime, src, dst, traffic, maxlat, sfcid] = tmp_request
                TD.AddRequest(req_idx, arrivaltime, src, dst, traffic, maxlat, sfcid)
            else:
                break
        for j in range(max_depls):
            tmp_deployment = deployment[j*n_depl_features:(j+1)*n_depl_features]
            if sum(tmp_deployment) != 0:
                [node_id, vnf_typenum, n_inst] = tmp_deployment
                TD.AddDeployment(node_id, vnf_typenum, n_inst)
            else:
                break
        for j in range(max_labels):
            tmp_label = label[j*n_label_features:(j+1)*n_label_features]
            if sum(tmp_label) != 0:
                [req_idx, vnf_typenum, node_id] = tmp_label
                TD.AddLabel(req_idx, vnf_typenum, node_id)

        if i % 1000 == 0:
            print("request : ", request)
            print("deployment : ", deployment)
            print("label : ", label)
            TD.PrintTopology(vnf=True, req=True)
            for req_idx in TD.reqs.keys():
                anno = TD.Generate_ANNO(req_idx)
            print("{}/{} are processed".format(i, dataset.__len__()))

    print("Done")
    
