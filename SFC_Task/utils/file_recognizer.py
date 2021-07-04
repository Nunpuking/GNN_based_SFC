import numpy as np
import pandas as pd

def TopologyRecognizer(topo_path, mode):
    if mode == 'Simulation':

        ''' Note about topology file format
        - INPUT
        0       line  : #_nodes #_edges
        #_nodes lines : node_id n_cpu
        #_edges lines : src_node dst_node capacity delay

        - OUTPUT
        n_nodes   : int
        n_edges   : int
        node_spec : { node_id : { 'cpu' : int }
                      ... }
        edge_spec : { (src_node(int), dst_node(int)) : { 'capacity' : int,
                                                         'delay'    : int }
                      ... }
        '''
        topology = np.array(pd.read_csv(topo_path, header=None).values.tolist())

        [n_nodes, n_edges] = topology[0].item().split()
        n_nodes, n_edges = int(n_nodes), int(n_edges)

        node_spec = {}
        for i in range(n_nodes):
            [node_id, n_cpu] = topology[i+1].item().split()
            node_id, n_cpu = int(node_id), int(n_cpu)
            node_spec[node_id] = { 'cpu' : n_cpu }

        edge_spec = {}
        for i in range(n_edges):
            [src, dst, capacity, delay] = topology[n_nodes+i+1].item().split()
            src, dst, capacity, delay = int(src), int(dst), int(capacity), int(delay)
            edge_spec[(src, dst)] = { 'capacity' : capacity, \
                                      'delay'    : delay }
            edge_spec[(dst, src)] = { 'capacity' : capacity, \
                                      'delay'    : delay }
    else:
        raise SyntaxError("ERROR: Wrong name for recognizer(pre-processing)")

    return n_nodes, n_edges, node_spec, edge_spec

def MiddleboxRecognizer(middlebox_path, mode):
    if mode == 'Simulation':
        ''' Note about middlebox file format
        - INPUT
        #_vnftypes lines : vnf_num, vnf_type, required_cpu, delay, capacity, dummy_value

        - OUTPUT
        vnf_spec     : { vnftype(str)  : { 'cpu'      : int,
                                           'delay'    : int,
                                           'capacity' : int, 
                                           'type_num' : int }
                    ... }
        vnf_spec_inv : { type_num(int) : { 'cpu'      : int,
                                           'delay'    : int,
                                           'capacity' : int, 
                                           'vnftype'  : str }
                    ... }
        '''
        middlebox = np.array(pd.read_csv(middlebox_path, header=None).values.tolist())

        vnf_spec = {}
        vnf_spec_inv = {}
        for i in range(len(middlebox)):
            [vnftype_num, vnftype, cpu, delay, capacity, _] = middlebox[i]
            vnftype_num, vnftype, cpu, delay, capacity = \
                        int(vnftype_num), str(vnftype), int(cpu), int(delay), int(capacity)
            vnf_spec[vnftype] = { 'cpu'      : cpu, \
                                  'delay'    : delay, \
                                  'capacity' : capacity,\
                                  'type_num' : vnftype_num }
            vnf_spec_inv[vnftype_num] = { 'cpu'      : cpu, \
                                          'delay'    : delay, \
                                          'capacity' : capacity,\
                                          'vnf_type' : vnftype }
    else:
        raise SyntaxError("ERROR: Wrong name for recognizer(pre-processing)")
    return vnf_spec, vnf_spec_inv

def SfctypesRecognizer(sfctypes_path, mode):
    if mode == 'Simulation':
        ''' Note about sfctypes file format
        - INPUT
        #_sfctypes lines : sfc_id, len(sfc), vnf1, vnf2, vnf3, vnf4

        - OUTPUT
        sfc_spec : { sfcid(int) : { 'length'    : int,
                                    'vnf_chain' : [str, str, str, str] }
                    ... }
        '''
        sfctypes = np.array(pd.read_csv(sfctypes_path, header=None).values.tolist())
        
        sfc_spec = {}
        for i in range(len(sfctypes)):
            [sfcid, length, vnf1, vnf2, vnf3, vnf4] = sfctypes[i]
            sfcid, length = int(sfcid), int(length)
            vnf1 = None if vnf1 == 'nan' else str(vnf1)
            vnf2 = None if vnf2 == 'nan' else str(vnf2)
            vnf3 = None if vnf3 == 'nan' else str(vnf3)
            vnf4 = None if vnf4 == 'nan' else str(vnf4)

            vnf_chain = [vnf for vnf in [vnf1,vnf2,vnf3,vnf4] if vnf]

            sfc_spec[sfcid] = { 'length' : length, \
                                'vnf_chain' : vnf_chain }
    else:
        raise SyntaxError("ERROR: Wrong name for recognizer(pre-processing)")
    return sfc_spec

#def request_recognizer(req_inst, mode): # Request preprocessor to update request list of TopologyDriver
    
