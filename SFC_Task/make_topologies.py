import os
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import copy

num_new_topo = 100

max_add_nodes = 12
p_n = 0.1

max_add_edges = 15
p_e = 0.3

max_remove_edges = 30
p_e_r = 0.3

def dijsktra(graph, initial, end):
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges_to[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.edges[(current_node, next_node)][0] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return []
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node

    path = path[::-1]
    return path[:]

class topology():
    def __init__(self, topo):
        self.n_nodes, self.n_edges = np.array(topo[0][0].split(), dtype=np.int)
        self.new_node_idx = self.n_nodes-1

        self.edges = {}
        self.edges_to = {}
        for i in range(self.n_edges):
            fromnode, tonode, capacity, delay =\
                    np.array(topo[i+1+self.n_nodes][0].split(), dtype=np.int)
            self.edges[(fromnode,tonode)] = (delay, capacity)
            self.edges[(tonode,fromnode)] = (delay, capacity)

            if fromnode not in self.edges_to.keys():
                self.edges_to[fromnode] = []
            if tonode not in self.edges_to.keys():
                self.edges_to[tonode] = []
            self.edges_to[fromnode].append(tonode)
            self.edges_to[tonode].append(fromnode)

    def add_nodes(self, max_add_nodes, p_n):
        for i in range(max_add_nodes):
            random_sample = np.random.rand(1)[0]
            if random_sample < p_n:
                self.n_nodes += 1
                self.new_node_idx += 1
                connect_node1 = np.random.choice(self.new_node_idx-1,1)[0]
                while(True):
                    connect_node2 = np.random.choice(self.new_node_idx-1,1)[0]
                    if connect_node2 != connect_node1:
                        break
                delay = np.random.choice(np.arange(20,151),1)[0]
                self.edges[(self.new_node_idx,connect_node1)] = (delay, 10000000)
                self.edges[(connect_node1,self.new_node_idx)] = (delay, 10000000)
                self.edges[(self.new_node_idx,connect_node2)] = (delay, 10000000)
                self.edges[(connect_node2,self.new_node_idx)] = (delay, 10000000)
                
                self.edges_to[self.new_node_idx] = []
                self.edges_to[self.new_node_idx].append(connect_node1)
                self.edges_to[self.new_node_idx].append(connect_node2)
                self.edges_to[connect_node1].append(self.new_node_idx)
                self.edges_to[connect_node2].append(self.new_node_idx)

        self.n_edges = int(len(self.edges.keys())/2.0)

    def add_edges(self, max_add_edges, p_e):
        for i in range(max_add_edges):
            random_sample = np.random.rand(1)[0]
            if random_sample < p_e:
                from_node = np.random.choice(self.new_node_idx,1)[0]
                while(True):
                    to_node = np.random.choice(self.new_node_idx,1)[0]
                    if to_node != from_node and (from_node,to_node) not in self.edges.keys():
                        break
                delay = np.random.choice(np.arange(20,151),1)[0]
                # You should check this edge is existed already
                self.edges[(from_node,to_node)] = (delay, 10000000)
                self.edges[(to_node,from_node)] = (delay, 10000000)

                self.edges_to[from_node].append(to_node)
                self.edges_to[to_node].append(from_node)

        self.n_edges = int(len(self.edges.keys())/2.0)

    def verify_topo(self):
        for from_node in self.edges_to.keys():
            for to_node in self.edges_to.keys():
                if from_node != to_node:
                    verify_path = dijsktra(self, from_node, to_node)
                    #print("from_node : ", from_node)
                    #print("to_node : ", to_node)
                    #print("verify_path : ", verify_path)
                    #print("len ver path : ", len(verify_path))
                    if len(verify_path) == 0:
                        return False
        return True

    def remove_edges(self, max_remove_edges, p_e_r):
        for i in range(max_remove_edges):
            random_sample = np.random.rand(1)[0]
            if random_sample < p_e_r:
                from_node = np.random.choice(self.new_node_idx,1)[0]
                if len(self.edges_to[from_node]) > 1:
                    to_node = np.random.choice(self.edges_to[from_node],1)[0]
                    if len(self.edges_to[to_node]) > 1:
                        org_edges_to = copy.deepcopy(self.edges_to)
                        org_edges = copy.deepcopy(self.edges)

                        self.edges_to[from_node].remove(to_node)
                        self.edges_to[to_node].remove(from_node)
                
                        del self.edges[(from_node,to_node)]
                        del self.edges[(to_node,from_node)]
                    
                        valid_check = self.verify_topo()
                        if valid_check == False:
                            self.edges_to = copy.deepcopy(org_edges_to)
                            self.edges = copy.deepcopy(org_edges)
                        #for node_id in self.edges_to.keys():
                        #    if len(self.edges_to[node_id]) == 1:
                        #        to_node = self.edges_to[node_id][0]
                        #        if len(self.edges_to[to_node]) == 1:
                        #            self.edges_to = org_edges_to.copy()
                        #            self.edges = org_edges.copy()


        self.n_edges = int(len(self.edges.keys())/2.0)

    def make_topo_file(self, new_topo_path):
        if os.path.exists(new_topo_path):
            os.remove(new_topo_path)
        new_topo = open(new_topo_path, 'a')
        
        new_topo.write("{} {}\n".format(self.n_nodes,self.n_edges))
        for node_id in range(self.n_nodes):
            new_topo.write("{} {}\n".format(node_id, 100))

        for from_node in range(self.n_nodes):
            for to_node in self.edges_to[from_node]:
                if to_node > from_node:
                    (delay, capacity) = self.edges[(from_node,to_node)]
                    new_topo.write("{} {} {} {}\n".format(from_node, to_node, capacity, delay))
        new_topo.close()
        


topo = pd.read_csv('../data/inet2', header=None).values.tolist()


for i in range(100,120):
    new_topo = topology(topo)
    new_topo.add_nodes(max_add_nodes, p_n)
    new_topo.add_edges(max_add_edges, p_e)
    new_topo.remove_edges(max_remove_edges, p_e_r)

    new_topo_path = '../data/new_topologies/inet2_' + str(i)
    new_topo.make_topo_file(new_topo_path)




