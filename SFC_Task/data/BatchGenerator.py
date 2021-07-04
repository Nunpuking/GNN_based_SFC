import numpy as np

def CheckPossibility(TDset, req_step, B):
    mask = np.zeros(B)

    for b in range(B):
        TD = TDset[b]
        if req_step >= len(TD.reqs.keys()):
            continue
        req_idx = list(TD.reqs.keys())[req_step]
        req = TD.reqs[req_idx]

        traffic = req['traffic']
        sfcid = req['sfcid']
        vnf_chain = TD.sfc_spec[sfcid]['vnf_chain']
        chain_length =TD.sfc_spec[sfcid]['length']
 
        tmp_flag = 1
        for vnf_type in vnf_chain:
            vnf_flag = False
            for (node_id, vnf_type), capacity in TD.vnfs.items():
                if capacity >= traffic:
                    vnf_flag = True
                    break
            if vnf_flag == False:
                tmp_flag = 0
                break
        mask[b] = tmp_flag
    return mask
 

def MakeEncodingBatch(TDset, req_step, B):
    '''
    B is the batch_size
    N is the number of nodes in the topology
    V is the number of VNF types in the network

    - INPUT
    TDset           : <B> class TD, set of TopologyDrivers
    req_step        : int         , time index of the current request in request sequences

    - OUTPUT
    annotation  : <B, N, 2+V> Array, Annotation matrices in the GG-NN layer
    A_out, A_in : <B, N, N>   Array, Adjacency matrices (outer&inner directions) in the GG-NN layer
    mask        : <B>         Array, Binary mask to indicate trainable ANNO, ADJ
    flag        : bool             , flag to indicate whether predictable requests are left or not

    '''

    N = TDset[0].n_nodes
    V = TDset[0].n_vnfs

    annotation = np.zeros((B, N, 2+V))
    A_out = np.zeros((B, N, N))
    A_in = np.zeros((B, N, N))
    mask = np.zeros(B)
    flag = False

    for b in range(B):
        TD = TDset[b]
        if req_step >= len(TD.reqs.keys()):
            continue
        req_idx = list(TD.reqs.keys())[req_step]

        tmp_anno = TD.Generate_ANNO(req_idx)
        tmp_A_out, tmp_A_in = TD.Generate_ADJ(req_idx)

        annotation[b] = tmp_anno
        A_out[b] = tmp_A_out
        A_in[b] = tmp_A_in
        mask[b] = 1
        flag = True

    return annotation, A_out, A_in, mask, flag

def MakeDecodingBatch(TDset, req_step, predict_mode, B):
    '''
    B is the batch size
    V is the number of VNF types in the network

    - INPUT
    TDset        : <B> class TD, set of TopologyDrivers
    req_step     : int         , time index of the current request in request sequences
    predict_mode : str         , 'NodeLevel' or 'VNFLevel'

    - OUTPUT
    from_node : <B>    int Array, indexes of current node
    vnf_now   : <B, V> int Array, indexes of the VNF type that is focused to process now
    vnf_all   : <B, V> int Array, indexes of the VNF types that are in the SFC chain
    mask      : <B, N> int Array, Binary mask to indicate trainable actions
    flag      : bool            , flag to indicate whether predictable requests are left or not

    '''
    V = TDset[0].n_vnfs
    N = TDset[0].n_nodes

    from_node = np.zeros(B)
    vnf_now = np.zeros((B, V))
    vnf_all = np.zeros((B, V))
    mask = np.zeros((B, N))
    flag = False

    for b in range(B):
        TD = TDset[b]
        if req_step >= len(TD.reqs.keys()):
            continue

        req_idx = list(TD.reqs.keys())[req_step]
        req = TD.reqs[req_idx]

        if req['complete'] == True:
            continue

        dst = req['dst']
        sfcid = req['sfcid']
        vnf_chain = TD.sfc_spec[sfcid]['vnf_chain']
        chain_length = TD.sfc_spec[sfcid]['length']

        tmp_from_node = req['node_gen'][-1]
        from_node[b] = tmp_from_node

        n_processed_vnfs = sum(req['vnf_gen'])
        if n_processed_vnfs < chain_length:
            vnf_now_type = vnf_chain[n_processed_vnfs]
            vnf_now_typenum = TD.vnf_spec[vnf_now_type]['type_num']
            vnf_now[b, vnf_now_typenum] = 1

            if predict_mode == 'VNFLevel':
                traffic = req['traffic']
                for (node_id, vnf_type), capacity in TD.vnfs.items():
                    if vnf_type == vnf_now_type and capacity >= traffic:
                        mask[b, node_id] = 1

        vnf_all_typenums = [TD.vnf_spec[tmp_vnf_type]['type_num']\
                             for tmp_vnf_type in vnf_chain]
        vnf_all[b, vnf_all_typenums] = 1

        if predict_mode == 'NodeLevel':
            nodes_to = TD.edges_to[tmp_from_node]
            mask[b, nodes_to] = 1
            mask[b, tmp_from_node] = 1

    flag = True if np.sum(mask) >= 1 else False

    return from_node, vnf_now, vnf_all, mask, flag

def MakeLabelBatch(TDset, req_step, gen_step, B):
    '''
    B is the batch size
    
    - INPUT
    TDset    : <B> class TD, set of TopologyDrivers
    req_step : int         , time index of the current request in request sequences
    gen_step : int         , time index of the current (node, vnf) generation 

    - OUTPUT
    label_node : <B> int Array, indexes of label nodes
    label_vnf  : <B> int Array, binary values of label vnf processings
    '''

    label_node = np.zeros(B)
    label_vnf = np.zeros(B)

    for b in range(B):
        TD = TDset[b]
        if req_step >= len(TD.reqs.keys()):
            continue

        req_idx = list(TD.reqs.keys())[req_step]
        req = TD.reqs[req_idx]

        if req['complete'] == True:
            continue

        label_node[b] = req['node_label'][gen_step]
        label_vnf[b] = req['vnf_label'][gen_step]

    return label_node, label_vnf

