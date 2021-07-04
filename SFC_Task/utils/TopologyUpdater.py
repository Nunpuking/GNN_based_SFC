import copy
import random

from utils.topology import TopologyDriver

def MakeTDset(batch_size, envrionment, predict_mode, temperature, recurrent_delay,\
                topo_path, sfctypes_path, middlebox_path):
    '''
    B  : batch_size

    - INPUT
    batch_size      : int
    envrionment     : str, 'Simulation' or 'OpenStack'
    predict_mode    : str, 'NodeLevel' or' VNFLevel'
    temperature     : float, the temperature factor for sotmax function
    recurrent_delay : int, the delay time of recurrent connections

    topo_path       : str, path of network topology
    sfctypes_path   : str, path of information of SFC types
    middlebox_path  : str, path of information of VNF types

    - OUTPUT
    TDset : <B>   class TD, set of TopologyDrivers 

    '''

    TDset = [0]*batch_size
    for b in range(batch_size):
        TDset[b] = TopologyDriver(envrionment, predict_mode, temperature, recurrent_delay)
        TDset[b].InitializeTopology(topo_path, middlebox_path, sfctypes_path)
    return TDset

def RandomTopology(TDset, topology_dir, sfctypes_path, middlebox_path):
    '''
    B : batch size

    - INPUT
    TDset           : <B> class TD, set of TopologyDrivers
    sfctypes_path   : str         , path of information of SFC types
    middlebox_path  : str         , path of information of VNF types

    '''
    B = len(TDset)

    random_topo_path = topology_dir + 'inet2_' + str(random.randint(0,100))

    for b in range(B):
        TDset[b].InitializeTopology(random_topo_path, middlebox_path, sfctypes_path)

def RandomDeployment(TDset, B):
    '''
    B : batch size

    - INPUT
    TDset : <B> class TD, set of TopologyDrivers

    '''

    N = TDset[0].n_nodes
    
    for b in range(B):
        TD = TDset[b]
        new_vnfs = {}
        for (node_id, vnf_type), capacity in TD.vnfs.items():
            random_node_id = random.randint(0,N-1)
            tmp_vnf_type = vnf_type
            tmp_capacity = capacity
            #new_vnfs[(random_node_id, tmp_vnf_type)] = tmp_capacity
            #_ = new_TD.vnfs.pop((node_id, vnf_type))

            if (random_node_id, tmp_vnf_type) in new_vnfs.keys():
                new_vnfs[(random_node_id, tmp_vnf_type)] += tmp_capacity
            else:
                new_vnfs[(random_node_id, tmp_vnf_type)] = tmp_capacity
        TDset[b].vnfs = copy.deepcopy(new_vnfs)

def SetTopology(TDset, requests, deployments, labels, data_spec, learning_mode):
    '''
    B  : batch_size
    MR : maximum number of requests in data sample
    MD : maximum number of deployments in data sample
    ML : maximum number of labels in data sample
    FR : the number of request features in data sample
    FD : the number of deployment features in data sample
    FL : the number of label features in data sample

    - INPUT
    TDset         : <B>   class TD, set of TopologyDrivers 
    requests      : <B, MR*FR> int, a batch of request samples
    deployments   : <B, MD*FD> int, a batch of deployment samples
    labels        : <B, ML*FL> int, a batch of label samples
    data_spec     : { 'max_reqs'         : MR,
                      'max_depls'        : MD,
                      'max_labels'       : ML,
                      'n_req_features'   : FR,
                      'n_depl_features'  : FD,
                      'n_label_features' : FL }
    learning_mode : str           , 'SL' or 'RL'

    '''
    def SubSetTopology(TD, instances, n_max, n_features, mode):
        for i in range(n_max):
            instance = instances[i*n_features:(i+1)*n_features]
            if sum(instance) == 0:
                break
            if mode == 'Request':
                [req_idx, arrivaltime, src, dst, traffic, maxlat, sfcid] = instance
                TD.AddRequest(req_idx, arrivaltime, src, dst, traffic, maxlat, sfcid)
            elif mode == 'Deployment':
                [node_id, vnf_typenum, n_inst] = instance
                TD.AddDeployment(node_id, vnf_typenum, n_inst)
            elif mode == 'Label':
                [req_idx, vnf_typenum, node_id] = instance
                TD.AddLabel(req_idx, vnf_typenum, node_id)


    B = len(requests)
    for b in range(B):
        TDset[b].ClearRequests()
        TDset[b].ClearDeployments()

        request, deployment, label = requests[b].tolist(),\
                                     deployments[b].tolist(),\
                                     labels[b].tolist()

        SubSetTopology(TDset[b], request, data_spec['max_reqs'],\
                         data_spec['n_req_features'], mode='Request')
        SubSetTopology(TDset[b], deployment, data_spec['max_depls'],\
                         data_spec['n_depl_features'], mode='Deployment')
        SubSetTopology(TDset[b], label, data_spec['max_labels'],\
                         data_spec['n_label_features'], mode='Label')
    return B

def UpdateGenerations(TDset, req_step, gen_nodes, B, gen_vnfs=None):
    '''
    B is the batch_size

    - INPUT
    TDset     : <B>  class TD, set of TopologyDrivers
    req_step  : int          , time index of the current request in request sequences
    gen_nodes : <B> int Array, indexes of the generated node actions
    gen_vnfs  : <B> int Array, binary values of the generated vnf processing actions
    
    '''

    for b in range(B):
        TD = TDset[b]
        if req_step >= len(TD.reqs.keys()):
            continue

        req_idx = list(TD.reqs.keys())[req_step]
        req = TD.reqs[req_idx]

        if req['complete'] == True:
            continue

        if gen_vnfs is not None:
            TD.UpdateGeneration(req_idx, gen_nodes[b], gen_vnfs[b])
        else:
            TD.UpdateGeneration(req_idx, gen_nodes[b], 1)

def UpdateCapacities(TDset, req_step, B):
    '''
    B is the batch_size

    - INPUT
    TDset     : <B>  class TD, set of TopologyDrivers
    req_step  : int          , time index of the current request in request sequences
    
    '''

    for b in range(B):
        TD = TDset[b]
        if req_step >= len(TD.reqs.keys()):
            continue

        req_idx = list(TD.reqs.keys())[req_step]
        req = TD.reqs[req_idx]

        if req['complete'] == False:
            continue
        gen_nodes = req['node_gen']
        gen_vnfs = req['vnf_gen']

        TD.UpdateCapacity(req_idx, gen_nodes, gen_vnfs)

