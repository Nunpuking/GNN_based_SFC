import torch
import copy
import numpy as np
from collections import OrderedDict

from utils.shortest_path import dijsktra
from utils.file_recognizer import TopologyRecognizer, MiddleboxRecognizer, SfctypesRecognizer

class TopologyDriver(): # Customized topology driver for operating with GG-RNN model
    def __init__(self, environment, predict_mode,\
                    temperature, recurrent_delay):
        # environment     : str  , 'OpenStack' or 'Simulation'
        # predict_mode    : str  , 'NodeLevel' or 'VNFLevel'
        # temperature     : float, the temperature factor for softmax function
        # recurrent_delay : int  , the delay time of recurrent connections
        self.environment = environment
        self.predict_mode = predict_mode
        self.temperature = temperature
        self.recurrent_delay = recurrent_delay
        
    def InitializeTopology(self, topo_path, middlebox_path, sfctypes_path):
        self.topo_path = topo_path
        self.n_nodes, self.n_edges, self.node_spec, self.edges =\
                 TopologyRecognizer(topo_path, self.environment)
        self.vnf_spec, self.vnf_spec_inv = MiddleboxRecognizer(middlebox_path, self.environment)
        self.n_vnfs = len(self.vnf_spec.keys())
        
        self.sfc_spec = SfctypesRecognizer(sfctypes_path, self.environment)

        # edges_to for label generation
        self.edges_to = {}
        for node_id in range(self.n_nodes):
            self.edges_to[node_id] = []
        for (from_node, to_node) in self.edges:
            self.edges_to[from_node].append(to_node)

        # create self connection edges
        for node_id in range(self.n_nodes):
            self.edges[(node_id, node_id)] = {'capacity':10000000, 'delay':0}

        self.ClearRequests()
        self.ClearDeployments()

    def ClearRequests(self):
        self.reqs = OrderedDict()

    def ClearDeployments(self):
        self.vnfs = {}

    def AddRequest(self, req_idx, arrivaltime, src, dst, traffic, maxlat, sfcid):
        '''
        - INPUT
        req_idx     : int, the unique index of the request among current request set
        arrivaltime : int, the arrival time of the request
        src         : int, the index of the source node
        dst         : int, the index of the destination node
        traffic     : int, the required traffic amount of the request
        maxlat      : int, the required maximum bound of processing time
        sfcid       : int, the type of SFC of the request

        - OUTPUT
        self.reqs   : { req_idx : { 'arrivaltime' : int
                                    'src'         : int
                                    'dst'         : int
                                    'traffic'     : int
                                    'maxlat'      : int
                                    'sfcid'       : int
                                    'node_label'  : int Array (optional)
                                    'vnf_label'   : int Array (optional)
                                    'node_gen'    : int Array (optional)
                                    'vnf_gen'     : int Array (optional)
                                    'complete'    : bool } }
        '''
        
        self.reqs[req_idx] = { 'arrivaltime' : arrivaltime, \
                               'src'         : src, \
                               'dst'         : dst, \
                               'traffic'     : traffic, \
                               'maxlat'      : maxlat, \
                               'sfcid'       : sfcid, \
                               'node_label'  : [], \
                               'vnf_label'   : [], \
                               'node_gen'    : [src], \
                               'vnf_gen'     : [0],
                               'complete'    : False }
   
    def AddDeployment(self, node_id, vnf_typenum, n_inst):
        '''
        - INPUT
        node_id     : int, the index of the deployed node
        vnf_typenum : int, the type number of the deployed VNF instance
        n_inst      : int, the number of deployed VNF instances on the node

        - OUTPUT
        self.vnfs : { (node_id, vnf_type) : int, the left capacity }

        '''

        vnf_type = self.vnf_spec_inv[vnf_typenum]['vnf_type']
        capacity = self.vnf_spec_inv[vnf_typenum]['capacity'] * n_inst
        self.vnfs[(node_id, vnf_type)] = capacity

    def AddLabel(self, req_idx, vnf_typenum, node_id):
        '''
        - INPUT
        req_idx     : int, the unique index of the request
        vnf_typenum : int, the type number of the label VNF instance
        node_id     : int, the index of the label VNF instance's deployed node
    
        - OUTPUT (if stored labels are reached to vnf_chain of the request)
        self.reqs[req_idx]['node_label'] : int Array
        self.reqs[req_idx]['vnf_label']  : int Array

        '''
        def GenerateLabel(topology, from_node, to_node, tmp_node_label, tmp_vnf_label, predict_mode):
            if predict_mode == 'NodeLevel':
                label_nodes = dijsktra(topology, from_node, to_node)
                if len(label_nodes) > 1:
                    label_nodes = label_nodes[1:]
                label_vnfs = [0]*(len(label_nodes)-1)
                label_vnfs += [1]
                tmp_node_label += label_nodes
                tmp_vnf_label += label_vnfs
            else:
                tmp_node_label += [to_node]
                tmp_vnf_label += [1]
            return tmp_node_label, tmp_vnf_label

        # Store recevied label instance
        vnf_type = self.vnf_spec_inv[vnf_typenum]['vnf_type']

        self.reqs[req_idx]['node_label'].append(node_id) # Store label node id temporarily
        self.reqs[req_idx]['vnf_label'].append(vnf_type) # Store label vnf type temporarily

        # Check the stored label instances are enough to generate label
        sfcid = self.reqs[req_idx]['sfcid']
        vnf_chain = self.sfc_spec[sfcid]['vnf_chain']

        generate_label = True
        label_chain = {}
        for vnf_type in vnf_chain:
            label_chain[vnf_type] = 0
            if vnf_type not in self.reqs[req_idx]['vnf_label']:
                generate_label = False

        # If it is enough, generate label
        if generate_label == True:
            node_ids = self.reqs[req_idx]['node_label']
            vnf_types = self.reqs[req_idx]['vnf_label']

            for (node_id, vnf_type) in zip(node_ids, vnf_types):
                label_chain[vnf_type] = node_id

            tmp_node_label = []
            tmp_vnf_label = []
            from_node = self.reqs[req_idx]['src']
            for vnf_type in vnf_chain:
                to_node = label_chain[vnf_type]
                tmp_node_label, tmp_vnf_label = GenerateLabel(self, from_node, to_node,\
                                             tmp_node_label, tmp_vnf_label, self.predict_mode)
                from_node = to_node
            to_node = self.reqs[req_idx]['dst']
            tmp_node_label, tmp_vnf_label = GenerateLabel(self, from_node, to_node,\
                                            tmp_node_label, tmp_vnf_label, self.predict_mode)
            tmp_vnf_label[-1] = 0
            self.reqs[req_idx]['node_label'] = tmp_node_label
            self.reqs[req_idx]['vnf_label'] = tmp_vnf_label

    def CapacityCheck(self, req_idx):
        traffic = self.reqs[req_idx]['traffic']
        sfcid = self.reqs[req_idx]['sfcid']

        vnf_chain = self.sfc_spec[sfcid]['vnf_chain']
        vnf_capacities = {}
        for vnf_type in vnf_chain:
            vnf_capacities[vnf_type] = 0
        
        for (node_id, vnf_type), capacity in self.vnfs.items():
            if vnf_type in vnf_chain:
                vnf_capacities[vnf_type] += capacity

        possibility = True
        for vnf_type in vnf_chain:
            if traffic > vnf_capacities[vnf_type]:
                possibility = False
                break
        return possibility

    def Generate_ADJ(self, req_idx):
        '''
        N is the number of nodes in the topology

        - INPUT
        req_idx         : int  , the unique index of the request

        - OUTPUT
        A_out : <N, N> Array, Adjacency matrix of outer direction
        A_in  : <N, N> Array, Adjacency matrix of inner direction
        '''

        def column_normalize(N, A, temperature, recurrent_delay):
            for col in range(N):
                A_col = A[:,col]
                vals = np.array([val for val in A_col if val != 0.0])
                mean = np.mean(vals)
                std = np.std(vals) if len(vals) > 1 else 1.0
                if std == 0.0 :
                    std = 1.0

                A_col = np.array([ (val-mean)/std+1e-10 if val != 0.0 else -float('inf')\
                                                                     for val in A_col ])

                z = A_col / temperature
                max_z = np.max(z)
                exp_z = np.exp(z-max_z)
                sum_exp_z = exp_z.sum()
                A_col = exp_z / sum_exp_z
         
                A_col = A_col * (1.0 - recurrent_delay)
                A_col[col] = recurrent_delay

                A[:,col] = A_col
            return A

        traffic = self.reqs[req_idx]['traffic']

        A_in = np.zeros((self.n_nodes,self.n_nodes))
        A_out = np.zeros((self.n_nodes,self.n_nodes))

        for from_node in range(self.n_nodes):
            for to_node in range(self.n_nodes):
                if from_node != to_node:
                    if (from_node,to_node) in self.edges.keys() and\
                        self.edges[(from_node,to_node)]['capacity'] >= traffic:
                        delay = self.edges[(from_node,to_node)]['delay']
                        A_out[from_node,to_node] = 1/delay
                    if (to_node,from_node) in self.edges.keys() and\
                        self.edges[(to_node,from_node)]['capacity'] >= traffic:
                        delay = self.edges[(to_node,from_node)]['delay']
                        A_in[to_node,from_node] = 1/delay
       
        A_out = column_normalize(self.n_nodes, A_out, self.temperature, self.recurrent_delay)
        A_in = column_normalize(self.n_nodes, A_in, self.temperature, self.recurrent_delay)

        return A_out, A_in

    def Generate_ANNO(self, req_idx):
        '''
        N is the number of nodes in the topology
        V is the number of VNF types in the network

        - INPUT
        req_idx : int, the unique index of the request

        - OUTPUT
        annotation : <N, V+2> Array, Annotation matrix in the GG-NN layer
        '''

        # Make Annotation matrix for model
        traffic = self.reqs[req_idx]['traffic']
        annotation = np.zeros((self.n_nodes,self.n_vnfs+2))
        src = self.reqs[req_idx]['src']
        dst = self.reqs[req_idx]['dst']
        annotation[src,0] = 1
        annotation[dst,-1] = 1

        for (node_id, vnf_type), capacity in self.vnfs.items():
            type_num = self.vnf_spec[vnf_type]['type_num']

            if capacity >= traffic:
                annotation[node_id, 1+type_num] = 1
        return annotation

    def UpdateCapacity(self, req_idx, gen_nodes, gen_vnfs):
        '''
        - INPUT
        req_idx   : int      , the unique index of the request
        gen_nodes : int Array, set of generated SFC path (node sequence)
        gen_vnfs  : int Array, set of generated VNF processings

        '''

        traffic = self.reqs[req_idx]['traffic']
        sfcid = self.reqs[req_idx]['sfcid']
        vnf_chain = self.sfc_spec[sfcid]['vnf_chain']
     
        gen_nodes = np.array(gen_nodes)
        gen_vnfs = np.array([1 if val == 1 else -1 for val in gen_vnfs])
        gen_vnf_nodes = [(val-1) for val in (gen_nodes+1)*gen_vnfs if val >= 0]

        for i, vnf_type in enumerate(vnf_chain):
            for (node_id, tmp_vnf_type), capacity in self.vnfs.items():
                if node_id == gen_vnf_nodes[i] and vnf_type == tmp_vnf_type:
                    if (capacity - traffic) < 0:
                        self.PrintTopology(vnf=True, req=True)
                        raise SyntaxError("{} request tried to process {} VNF on {} node"\
                                .format(req_idx, tmp_vnf_type, node_id))
                    self.vnfs[(node_id, tmp_vnf_type)] = capacity - traffic


    def UpdateGeneration(self, req_idx, gen_node, gen_vnf):
        '''
        - INPUT
        req_idx  : int, the unique index of the request
        gen_node : int, the index of generated node
        gen_vnf  : int, binary flag indicate the VNF process action

        '''
        if self.reqs[req_idx]['complete'] == False:
            # Update the generated nodes
            self.reqs[req_idx]['node_gen'].append(gen_node)

            # Update the generated VNF processing
            vnf_process_flag = False
            traffic = self.reqs[req_idx]['traffic']
            sfcid = self.reqs[req_idx]['sfcid']
            vnf_chain = self.sfc_spec[sfcid]['vnf_chain']
            n_processed_vnfs = sum(self.reqs[req_idx]['vnf_gen'])

            if gen_vnf == 1 and n_processed_vnfs < len(vnf_chain):
                vnf_now_type = vnf_chain[n_processed_vnfs]

                for (node_id, vnf_type), capacity in self.vnfs.items():
                    if node_id == gen_node and vnf_type == vnf_now_type and capacity >= traffic:
                        vnf_process_flag = True
            if vnf_process_flag == True:
                self.reqs[req_idx]['vnf_gen'].append(1)
            else:
                self.reqs[req_idx]['vnf_gen'].append(0)
            
            n_processed_vnfs = sum(self.reqs[req_idx]['vnf_gen'])
            if self.predict_mode == 'VNFLevel' and n_processed_vnfs == len(vnf_chain):
                self.reqs[req_idx]['complete'] = True
                self.reqs[req_idx]['node_gen'].append(self.reqs[req_idx]['dst'])
                self.reqs[req_idx]['vnf_gen'].append(0)
            
            dst = self.reqs[req_idx]['dst']
            if self.predict_mode == 'NodeLevel' and n_processed_vnfs == len(vnf_chain)\
                 and gen_node == dst:
                self.reqs[req_idx]['complete'] = True

    def ComputeReward(self, req_idx, delay_coeff, reward_mode):
        '''
        !!! This method only computes the reward based on current generated nodes and
             vnf processing action sequences. Therefore 'UpdateGeneration' firstly.

        - INPUT
        req_idx     : int  , the unique index of the request
        delay_coeff : float, the controlling parameter of delay reward
        reward_mode : str  , reward type ('REINFORCE', )

        - OUTPUT
        reward : float

        '''
        def REINFORCE_reward(TD, req_idx, success_reward, delay_coeff):
            if TD.reqs[req_idx]['complete'] == True:
                delay = TD.ComputeDelay(req_idx, 'Generation', TD.predict_mode)
                reward = success_reward - delay_coeff*delay
            else:
                reward = 0
            return reward
        
        success_reward = 10000

        if reward_mode == 'REINFORCE':
            reward = REINFORCE_reward(self, req_idx, success_reward, delay_coeff)
        else:
            raise SyntaxError('Wrong Reward Mode name in TD.ComputeReward method')
        
        return reward

    def ComputeDelay(self, req_idx, mode, predict_mode):
        '''
        !!! This method computes total delay of generated path (also label path if it is existed).
            Therefore, complete path first.

        - INPUT
        req_idx      : int, the unique index of the request
        mode         : str, 'Label' or 'Generation' which one for compute the delay
        predict_mode : str, 'NodeLevel' or 'VNFLevel'

        - OUTPUT
        delay : int

        '''
        def VNFs2Path(topology, nodes):
            path = [nodes[0]]
            vnf_seq = [0]

            from_node = None
            for idx, to_node in enumerate(nodes):
                if from_node is not None:
                    tmp_nodes = dijsktra(topology, from_node, to_node)
                    if len(tmp_nodes) > 1:
                        tmp_nodes = tmp_nodes[1:]
                    path += tmp_nodes
                    vnf_seq += [0]*(len(tmp_nodes)-1) + [1]
                from_node = to_node
            vnf_seq[-1] = 0

            return path, vnf_seq
        
        delay = 0

        req = self.reqs[req_idx]
        vnf_chain = self.sfc_spec[req['sfcid']]['vnf_chain']

        if mode == 'Generation':
            nodes = req['node_gen']
            vnfs = req['vnf_gen']
        else:
            nodes = req['node_label']
            vnfs = req['vnf_label']

            nodes.insert(0, req['src'])
            vnfs.insert(0, 0)

        if predict_mode == 'VNFLevel':
            nodes, vnfs = VNFs2Path(self, nodes)

        from_node = None
        n_processed_vnfs = 0
        for i, (node, vnf) in enumerate(zip(nodes, vnfs)):
            delay += self.edges[(from_node, node)]['delay'] if from_node is not None else 0
            from_node = node
            if vnf == 1:
                process_vnftype = vnf_chain[n_processed_vnfs]
                self.vnf_spec[process_vnftype]['delay']
                n_processed_vnfs += 1

        return delay

    def PrintTopology(self,\
                        topology=False,\
                        edge=False,\
                        vnf=False,\
                        req=False):
        if topology == True:
            print("-------------Topology Basic Info.--------------")
            print("Number of nodes : {}".format(self.n_nodes))
            print("Number of edges : {}".format(self.n_edges))
            print("Number of VNFs  : {}".format(self.n_vnfs))
            print("Topology type   : {}".format(self.topo_path))
            print()
        if edge == True:
            print("-------------Edges--------------")
            for edge_id in self.edges.keys():
                print("{} edge : {} delay / {} capacity".format(edge_id,\
                                            self.edges[edge_id]['delay'],\
                                            self.edges[edge_id]['capacity']))
            print()
        if vnf == True:
            print("-------------VNFs--------------")
            for vnf_type in self.vnf_spec.keys():
                print("{} VNF TYPE".format(vnf_type))
                for (node_id, tmp_vnf_type) in self.vnfs.keys():
                    if tmp_vnf_type == vnf_type:
                        print("------{} node, {} capacity".format(node_id,\
                                                        self.vnfs[(node_id,tmp_vnf_type)]))
            print()
        if req == True:
            print("-------------REQs---------------")
            for req_idx in self.reqs.keys():
                print("{} Request\t: {}".format(req_idx, self.reqs[req_idx]))
            print()

 
        

        

