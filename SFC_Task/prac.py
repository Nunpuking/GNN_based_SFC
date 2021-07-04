import numpy as np
import pandas as pd
from collections import OrderedDict
import random
import torch
import torch.nn as nn
import time
import copy

from trainer.tmp import Hello
#from topology import TopologyDriver

label_node = np.zeros(5)

label_node[1] = 1

print(label_node)

label_node = np.array(label_node, dtype=np.int)
print(label_node)

'''
self.link_costs = {}
self.link_costs[(0,0)] = 0
self.link_costs[(0,1)], self.link_costs[(1,0)] = 1, 1
self.link_costs[(0,2)], self.link_costs[(2,0)] = 85, 85
self.link_costs[(0,3)], self.link_costs[(3,0)] = 176, 176
self.link_costs[(0,4)], self.link_costs[(4,0)] = 118, 118
self.link_costs[(0,5)], self.link_costs[(5,0)] = 59, 59
self.link_costs[(0,6)], self.link_costs[(6,0)] = 113, 113
self.link_costs[(0,7)], self.link_costs[(7,0)] = 307, 307
self.link_costs[(0,8)], self.link_costs[(8,0)] = 108, 108
self.link_costs[(0,9)], self.link_costs[(9,0)] = 305, 305
self.link_costs[(0,10)], self.link_costs[(10,0)] = 385, 385
self.link_costs[(0,11)], self.link_costs[(11,0)] = 85, 85

self.link_costs[(1,1)] = 0
self.link_costs[(1,2)], self.link_costs[(2,1)] = 84, 84
self.link_costs[(1,3)], self.link_costs[(3,1)] = 175, 175
self.link_costs[(1,4)], self.link_costs[(4,1)] = 117, 117
self.link_costs[(1,5)], self.link_costs[(5,1)] = 58, 58
self.link_costs[(1,6)], self.link_costs[(6,1)] = 112, 112
self.link_costs[(1,7)], self.link_costs[(7,1)] = 306, 306
self.link_costs[(1,8)], self.link_costs[(8,1)] = 107, 107
self.link_costs[(1,9)], self.link_costs[(9,1)] = 304, 304
self.link_costs[(1,10)], self.link_costs[(10,1)] = 384, 384
self.link_costs[(1,11)], self.link_costs[(11,1)] = 84, 84

self.link_costs[(2,2)] = 0
self.link_costs[(2,3)], self.link_costs[(3,2)] = 143, 143
self.link_costs[(2,4)], self.link_costs[(4,2)] = 170, 170
self.link_costs[(2,5)], self.link_costs[(5,2)] = 26, 26
self.link_costs[(2,6)], self.link_costs[(6,2)] = 80, 80
self.link_costs[(2,7)], self.link_costs[(7,2)] = 359, 359
self.link_costs[(2,8)], self.link_costs[(8,2)] = 70, 70
self.link_costs[(2,9)], self.link_costs[(9,2)] = 272, 272
self.link_costs[(2,10)], self.link_costs[(10,2)] = 352, 352
self.link_costs[(2,11)], self.link_costs[(11,2)] = 93, 93

self.link_costs[(3,3)] = 0
self.link_costs[(3,4)], self.link_costs[(4,3)] = 153, 153
self.link_costs[(3,5)], self.link_costs[(5,3)] = 117, 117
self.link_costs[(3,6)], self.link_costs[(6,3)] = 63, 63
self.link_costs[(3,7)], self.link_costs[(7,3)] = 162, 162
self.link_costs[(3,8)], self.link_costs[(8,3)] = 213, 213
self.link_costs[(3,9)], self.link_costs[(9,3)] = 129, 129
self.link_costs[(3,10)], self.link_costs[(10,3)] = 209, 209
self.link_costs[(3,11)], self.link_costs[(11,3)] = 236, 236

self.link_costs[(4,4)] = 0
self.link_costs[(4,5)], self.link_costs[(5,4)] = 144, 144
self.link_costs[(4,6)], self.link_costs[(6,4)] = 90, 90
self.link_costs[(4,7)], self.link_costs[(7,4)] = 189, 189
self.link_costs[(4,8)], self.link_costs[(8,4)] = 240, 240
self.link_costs[(4,9)], self.link_costs[(9,4)] = 225, 225
self.link_costs[(4,10)], self.link_costs[(10,4)] = 311, 311
self.link_costs[(4,11)], self.link_costs[(11,4)] = 201, 201

self.link_costs[(5,5)] = 0
self.link_costs[(5,6)], self.link_costs[(6,5)] = 54, 54
self.link_costs[(5,7)], self.link_costs[(7,5)] = 333, 333
self.link_costs[(5,8)], self.link_costs[(8,5)] = 96, 96
self.link_costs[(5,9)], self.link_costs[(9,5)] = 246, 246
self.link_costs[(5,10)], self.link_costs[(10,5)] = 326, 326
self.link_costs[(5,11)], self.link_costs[(11,5)] = 119, 119

self.link_costs[(6,6)] = 0
self.link_costs[(6,7)], self.link_costs[(7,6)] = 279, 279
self.link_costs[(6,8)], self.link_costs[(8,6)] = 150, 150
self.link_costs[(6,9)], self.link_costs[(9,6)] = 192, 192
self.link_costs[(6,10)], self.link_costs[(10,6)] = 272, 272
self.link_costs[(6,11)], self.link_costs[(11,6)] = 173, 173

self.link_costs[(7,7)] = 0
self.link_costs[(7,8)], self.link_costs[(8,7)] = 413, 413
self.link_costs[(7,9)], self.link_costs[(9,7)] = 36, 36
self.link_costs[(7,10)], self.link_costs[(10,7)] = 122, 122
self.link_costs[(7,11)], self.link_costs[(11,7)] = 390, 390

self.link_costs[(8,8)] = 0
self.link_costs[(8,9)], self.link_costs[(9,8)] = 342, 342
self.link_costs[(8,10)], self.link_costs[(10,8)] = 422, 422
self.link_costs[(8,11)], self.link_costs[(11,8)] = 23, 23

self.link_costs[(9,9)] = 0
self.link_costs[(9,10)], self.link_costs[(10,9)] = 86, 86
self.link_costs[(9,11)], self.link_costs[(11,9)] = 365, 365

self.link_costs[(10,10)] = 0
self.link_costs[(10,11)], self.link_costs[(11,10)] = 445, 45

self.link_costs[(11,11)] = 0
'''
