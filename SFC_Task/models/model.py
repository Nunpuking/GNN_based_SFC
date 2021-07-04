import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def Call_Model(args, n_vnfs):
    # Prepare configurations
    anno_dim = n_vnfs + 2
    
    # Call model
    if args.model_name == 'GG_RNN':
        model = GG_RNN(args.max_n_nodes, n_vnfs,\
                        args.node_state_dim, args.GRU_steps, anno_dim, args.posenc_node_dim,\
                        args.vnf_dim, args.device, args.predict_mode)
    elif args.model_name == 'DNN':
        model = DNN(args, n_nodes, n_vnfs)
    else:
        raise SyntaxError("ERROR: {} is Wrong model name".format(args.model_name)) 
    return model

class Embedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(input_dim, embedding_dim).type(torch.float))

    def forward(self, x):
        return torch.matmul(x, self.param)

def position_encoding_init(n_position, emb_dim):
    position_enc = np.array([\
            [pos / np.power(10000, 2*(j // 2) / emb_dim) for j in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

class GG_RNN(nn.Module):
    def __init__(self, max_N, V,\
                 E, T, anno_dim, node_dim, vnf_dim,\
                 device, mode):
        super(GG_RNN, self).__init__()

        '''
        - INPUT
        N        : int, the number of nodes in the topology
        max_N    : int, the maximum number of nodes (it determines position-encoding values)
        V        : int, the number of VNF types in the network

        E        : int, the dimension of a node representation
        T        : int, the number of recurrent steps in GG-NN layer
        anno_dim : int, the dimension of embedded annotatoin vector in GG-NN layer
        node_dim : int, the dimension of position-encoding
        vnf_dim  : int, the dimension of embedded VNF info. vector

        device   : str, the name of computing device
        mode     : str, mode of the model 'NodeLevel' or 'VNFLevel'
        '''

        self.A = 3
        self.V = V
        self.T = T
        self.E = E
        self.anno_dim = anno_dim
        self.vnf_dim = vnf_dim
        self.node_dim = node_dim

        self.device = device
        self.mode = mode

        # Encoder
        self.anno_emb = Embedding(anno_dim, E)
        self.GRUcell = nn.GRUCell(2*E, E, bias=False)

        # Decoder
        self.pos_enc = position_encoding_init(max_N, node_dim)
        self.pos_enc = Variable(self.pos_enc).to(device)

        self.vnf_now_emb = Embedding(V, vnf_dim)
        self.vnf_all_emb = Embedding(V, vnf_dim)
        self.decoder_GRUcell = nn.GRUCell(E + node_dim + 2*vnf_dim, 2*E)

        self.decoder_out = nn.Sequential(\
                    nn.Linear(2*E, E),\
                    nn.ReLU(True),\
                    nn.Linear(E, self.A))

        # Probability
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def Adjacency_Eye(self, Adj, B, N):
        EyeAdj = torch.zeros(B*N, B*N)
        for b in range(B):
            EyeAdj[b*N:(b+1)*N,b*N:(b+1)*N] = Adj[b]
        
        return EyeAdj

    def encoder(self, annotation, A_out, A_in):
        '''
        B is the batch size
        N is the number of nodes in the topology
        V is the number of VNF types in the network
        E is the dimension of a node representation
        
        - INPUT
        annotation  : <B, N, 2+V> Array, Annotation matrices in the GG-NN layer
        A_out, A_in : <B, N, N>   Array, Adjacency matrices (out&in directions) in the GG-NN layer
        
        - OUTPUT
        enc_out : <B*N, E> Tensor, encoded representations of nodes

        '''
        B, N, _ = A_out.shape
        self.B = B
        self.N = N

        A_out = self.Adjacency_Eye(torch.from_numpy(A_out).type(torch.float), B, N)
        A_in = self.Adjacency_Eye(torch.from_numpy(A_in).type(torch.float), B, N)

        annotation = Variable(torch.from_numpy(annotation).type(torch.float)).to(self.device)
        A_out, A_in = Variable(A_out).to(self.device), Variable(A_in).to(self.device)

        annotation = annotation.reshape(B*N, self.anno_dim)

        # Annotation Embedding
        h = self.anno_emb(annotation) # <B*N, E>
       
        # GG-NN Layer
        for i in range(self.T):
            a_out = torch.matmul(A_out,h)
            a_in = torch.matmul(A_in,h)
            a = torch.cat((a_out,a_in), dim=1)
            h = self.GRUcell(a,h)
        enc_out = h # <B*N, E>
        return enc_out

    def forward(self, enc_out, from_node, vnf_now, vnf_all, mask, h=None):
        '''
        B is the batch size
        N is the number of nodes in the topology
        V is the number of VNF types in the network
        E is the dimension of a node representation
        
        - INPUT
        enc_out   : <B*N, E>  Tensor, encoded representations of nodes
        from_node : <B>          int, indexes of current node
        vnf_now   : <B, V>       int, indexes of the VNF type that is focused to process now
        vnf_all   : <B, V>       int, indexes of the VNF types that are in the SFC chain
        mask      : <B*N>        int, binary mask indicate trainable actions
        h         : <B*N, 2E> Tensor, previous hidden state

        - OUTPUT
        node_logits : <B, N>     Tensor, final logits of node actions
        vnf_logits  : <B, N, 2>  Tensor, final logits of vnf processing actions
        hidden      : <B*N, 2E> Tensor, current hidden state
        '''

        # VNF info Embeddings
        vnf_now = Variable(torch.from_numpy(vnf_now).type(torch.float)).to(self.device)
        vnf_all = Variable(torch.from_numpy(vnf_all).type(torch.float)).to(self.device)

        vnf_now = self.vnf_now_emb(vnf_now).repeat(1,self.N).reshape(self.B*self.N, self.vnf_dim)
        vnf_all = self.vnf_all_emb(vnf_all).repeat(1,self.N).reshape(self.B*self.N, self.vnf_dim)

        # Node encoding
        node_pos_enc = self.pos_enc[from_node].repeat(1,self.N).reshape(self.B*self.N, self.node_dim)
        
        # Decoding Layer
        concat_input = torch.cat((enc_out, node_pos_enc, vnf_all, vnf_now),1)

        if h is None:
            h = Variable(torch.zeros([self.B*self.N, 2*self.E])).to(self.device)

        mask = np.array([1 if val == 1 else 1e10 for val in mask])
        mask = Variable(torch.from_numpy(mask).type(torch.float)).to(self.device)

        mask = mask.reshape(self.B,self.N)
        mask = mask.unsqueeze(2).repeat(1,1,self.A)
        
        hidden = self.decoder_GRUcell(concat_input, h)
        output = self.decoder_out(hidden)

        output = output.reshape(self.B,self.N,-1)
        max_values = torch.max(torch.max(output,2).values,1).values.unsqueeze(1)\
                    .repeat(1,self.N*self.A).reshape(self.B,self.N,-1)

        output -= max_values + 1

        output = (mask*output).reshape(self.B*self.N,-1)
        output += 1
 
        node_logits = output[:,0].reshape(self.B, -1)
        vnf_logits = output[:,1:].reshape(self.B, self.N, -1)
        return node_logits, vnf_logits, hidden

    def Load_PTmodel(self, pt_model):
        self.anno_emb = pt_model.anno_emb
        self.GRUcell = pt_model.GRUcell
        self.vnf_all_emb = pt_model.vnf_all_emb
        self.vnf_now_emb = pt_model.vnf_now_emb
        self.decoder_GRUcell = pt_model.decoder_GRUcell
        self.decoder_out = pt_model.decoder_out

    '''
    def forward(self, Anno, A_out, A_in, vnf_chain, node_now, enc_out=None):
        
        N is the number of nodes in the topology ( = self.n_nodes)
        V is the number of VNF types in the network ( = self.n_vnfs)
        E is the dimension of a node representation
        A is the number of actions
        
        - INPUT
        Anno          : <N, 2+V >    Array, Annotaton matrix in the GG-NN layer
        A_out, A_in   : <N, N >      Array, Adjacency matrices (out&in directions) 
                                            in the GG-NN layer     
        vnf_chain     : int Array         , indexes of the VNF types that are in the SFC chain
        node_now      : int               , index of the current node
        
        (optional)
        vnf_type_nums : { str : int } dict, dictionary for vnf_type(str)-type_num(int) matching
        node_label    : int Array         , indexes of the nodes in the label 
                                            for each node traversing action
        process_label : int Array         , 0(not process) or 1(process) 
                                            for each VNF processing action
        vnf_mask      : <N> int Array     , 0(mask) or 1(non-mask) for each nodes (VNF instances)
                                            only for 'VNF-level' prediction tasks
        enc_out       : <N, E>      Tensor, encoded node representatoins of the current request

        - OUTPUT
        probs : <N, S> Array, probabilities of actions
        

        if enc_out is not None:
            Anno = Variable(torch.from_numpy(Anno).type(torch.long)).to(self.device)
            A_out = Variable(torch.from_numpy(A_out).type(torch.float)).to(self.device)
            A_in = Variable(torch.from_numpy(A_in).type(torch.float)).to(self.device)

            enc_out = self.encoder(Anno, A_out, A_in)

        
        hidden = None

        loss = 0
        corrects = 0

        vnf_all_nums = [vnf_type_nums[vnf_type] for vnf_type in vnf_chain if vnf_type != None]
        gen_vnf_inst = torch.zeros(self.n_vnfs)
        for vnf_type in vnf_chain:
            if vnf_type is None:
                break
            vnf_now_num = vnf_type_nums[vnf_type]
            dec_out, hidden = self.decoder(enc_out, node_now, vnf_now_num, vnf_all_nums, hidden) # <N, 1>

            tmp_mask = vnf_mask[vnf_now_num] # <N, >
            dec_out_max = torch.max(dec_out)
            dec_out = torch.transpose(torch.mul(tmp_mask.unsqueeze(0).T, (dec_out-dec_out_max)),0,1)
            tmp_label = label[vnf_now_num]

            loss += self.criterion(dec_out, tmp_label.unsqueeze(0))
            probs = self.softmax(dec_out)
            #print("dec_out : ", dec_out)
            #print("probs : ", probs)

            pred = torch.argmax(probs).item()

            gen_vnf_inst[vnf_now_num] = pred
            corrects += 1 if pred == tmp_label.item() else 0
            if mode == 'train':
                node_now = tmp_label.item()
            else:
                node_now = pred
        return loss, gen_vnf_inst, corrects
    '''

class DNN(nn.Module):
    def __init__(self, args, n_nodes, n_vnfs):
        super(DNN, self).__init__()
        self.args = args
        self.device = args.device
        self.GRU_steps = args.GRU_steps

        self.n_nodes = n_nodes
        self.n_vnfs = n_vnfs
        self.annotation_dim = n_vnfs + 2

        self.anno_emb = Embedding(self.annotation_dim, args.node_state_dim)
        #self.GRUcell = nn.GRUCell(2*args.node_state_dim, args.node_state_dim, bias=False)

        self.pos_enc = position_encoding_init(args.max_n_nodes, args.posenc_node_dim)
        self.pos_enc = Variable(self.pos_enc).to(args.device)

        self.vnf_all_emb = Embedding(self.n_vnfs, args.vnf_dim)
        self.vnf_now_emb = Embedding(self.n_vnfs, args.vnf_dim)

        #self.decoder_GRUcell = nn.GRUCell(\
        #            args.node_state_dim + args.posenc_node_dim, args.node_state_dim*2)
        self.decoder_out1 = nn.Sequential(\
                        nn.Linear(args.node_state_dim + args.posenc_node_dim + 2*args.vnf_dim \
                        + 2*self.n_nodes*self.n_nodes, args.node_state_dim*5),\
                        nn.ReLU(True))

        self.decoder_out2 = nn.Sequential(\
                    nn.Linear(args.node_state_dim*5, args.node_state_dim*3),\
                    nn.ReLU(True),\
                    nn.Linear(args.node_state_dim*3, args.node_state_dim),\
                    nn.ReLU(True),\
                    nn.Linear(args.node_state_dim,1))

        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    ''''
    def encoder(self, annotation, A_out, A_in):
        h = self.anno_emb(annotation) # <N, E>
        for i in range(self.GRU_steps):
            a_out = torch.matmul(A_out,h)
            a_in = torch.matmul(A_in,h)
            a = torch.cat((a_out,a_in), dim=1)
            h = self.GRUcell(a,h)
        enc_out = h
        return enc_out
    '''
    def decoder(self, concat_input, from_node, vnf_now_num, vnf_all_nums):
        vnf_all = Variable(torch.zeros(self.n_vnfs)).to(self.device)
        for type_num in vnf_all_nums:
            vnf_all[type_num] = 1
        vnf_all = self.vnf_all_emb(vnf_all).repeat(self.n_nodes,1)
        
        vnf_now = Variable(torch.zeros(self.n_vnfs)).to(self.device)
        vnf_now[vnf_now_num] = 1
        vnf_now = self.vnf_now_emb(vnf_now).repeat(self.n_nodes,1)
   
        node_pos_enc = self.pos_enc[from_node].repeat(self.n_nodes,1)
        concat_input = torch.cat((concat_input, node_pos_enc, vnf_all, vnf_now),1)

        hidden = self.decoder_out1(concat_input)
        output = self.decoder_out2(hidden)
        return output, hidden
    
    def forward(self, A_out, A_in, Anno, vnf_chain, src, vnf_mask,\
                 vnf_type_nums, label, mode='train'):
        
        A_out, A_in, Anno, vnf_mask = Variable(A_out).to(self.device),\
                                        Variable(A_in).to(self.device),\
                                        Variable(Anno).to(self.device),\
                                        Variable(vnf_mask).to(self.device)
        label = Variable(label).to(self.device)

        #enc_out = self.encoder(Anno, A_out, A_in)
        Anno_emb = self.anno_emb(Anno) # <N, E>
        A_out = A_out.view(-1).repeat(self.n_nodes,1)
        A_in = A_in.view(-1).repeat(self.n_nodes,1)
        concat_input = torch.cat((Anno_emb, A_out, A_in), 1)

        node_now = src
        hidden = None
        loss = 0
        corrects = 0

        vnf_all_nums = [vnf_type_nums[vnf_type] for vnf_type in vnf_chain if vnf_type != None]
        gen_vnf_inst = torch.zeros(self.n_vnfs)
        for vnf_type in vnf_chain:
            if vnf_type is None:
                break
            vnf_now_num = vnf_type_nums[vnf_type]
            dec_out, hidden = self.decoder(concat_input, node_now, vnf_now_num, vnf_all_nums) #<N, 1>

            tmp_mask = vnf_mask[vnf_now_num] # <N, >
            dec_out_max = torch.max(dec_out)
            dec_out = torch.transpose(torch.mul(tmp_mask.unsqueeze(0).T, (dec_out-dec_out_max)),0,1)
            tmp_label = label[vnf_now_num]

            loss += self.criterion(dec_out, tmp_label.unsqueeze(0))
            probs = self.softmax(dec_out)

            pred = torch.argmax(probs).item()

            gen_vnf_inst[vnf_now_num] = pred
            corrects += 1 if pred == tmp_label.item() else 0
            if mode == 'train':
                node_now = tmp_label.item()
            else:
                node_now = pred
            '''
            if loss.item() > 10:
                print("label : ", label)
                print("Anno : ", Anno)
                print(vnf_type)
                print("tmp_mask : ")
                print(tmp_mask)
                print("masked_out ")
                print(dec_out)
                print("probs : ", probs)
                print("loss : ", loss)

                print("pred : ", pred)
                print("corrects : ", corrects)
            '''
        return loss, gen_vnf_inst, corrects

    def Load_PTmodel(self, pt_model):
        self.anno_emb = pt_model.anno_emb
        self.GRUcell = pt_model.GRUcell
        self.decoder_GRUcell = pt_model.decoder_GRUcell
        self.decoder_out = pt_model.decoder_out
