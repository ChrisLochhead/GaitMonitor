'''
This file concerns all Machine learning networks related to the ST-GCN
'''
#imports
import torch
torch.manual_seed(42)
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool,  GATv2Conv, ChebConv
from torch.nn import Linear, BatchNorm1d, ReLU
import torch.nn as nn
from torch_geometric.nn.conv import GraphConv, AGNNConv
from torchvision.models import resnet18

#dependencies
from Programs.Machine_Learning.GCN.Dataset_Obj import *
from Programs.Machine_Learning.GCN.Modules import ST_GCN_Block, STJA_GCN_Block, Cust_STGCN_Block, Gait_Graph2_Block, ST_TAGCN_Block, VAE_ST_TAGCN_Block
from Programs.Machine_Learning.GCN.GCN import S_GCN, T_GCN


class GCN_GraphNetwork(torch.nn.Module):
    def __init__(self, dim_in, dim_h, num_classes, n_inputs, data_dims, batch_size, hcf = False, stgcn_size = 2, stgcn_filters = [128,128], 
                 max_cycle = 7, num_nodes_per_graph = 18, device = 'cuda', type = 'ST_TAGCN_Block'):
        super(GCN_GraphNetwork, self).__init__()

        dim_half = int(dim_h/2)
        dim_4th = int(dim_half/2)
        dim_8th = int(dim_4th/2)
        self.dim_in = dim_in
        self.num_inputs = n_inputs
        self.hcf = hcf
        self.size_stgcn = stgcn_size
        self.stgcn_filters = stgcn_filters
        self.batch_size = batch_size
        self.data_dims = data_dims
        self.num_nodes_per_graph = num_nodes_per_graph
        self.cycle_size = max_cycle * num_nodes_per_graph 
        self.device = device
        #0 = GAT, 1=ST-GCN, 2=ST-AGCN
        self.model_type = type

        #Two sub-streams: the first contains the sequences that each input stream goes through,
        #the second is the single combined stream 
        self.streams = []
        for i in range(self.num_inputs):
            #if HCF is activated, it will always be the last dataset in the number of inputs, so add a HCF sequence to the
            #end of the stream stack.
            if i == len(range(self.num_inputs)) - 1 and hcf == True :
                 # This is a normal linear net for HCF data without graphical structure
                self.streams.append(torch.nn.Sequential(
                    Linear(self.dim_in[-1], dim_h), ReLU(), BatchNorm1d(dim_h),
                    Linear(dim_h, dim_half),ReLU(),BatchNorm1d(dim_half),
                    Linear(dim_half, dim_4th),ReLU(),BatchNorm1d(dim_4th),
                    Linear(dim_4th, dim_8th)
                ).to(device))

            else:
                #ST_GCN_Block, STJA_GCN_Block, Cust_STGCN_Block, Gait_Graph2_Block, ST_TAGCN_Block
                i_stream = []
                if self.model_type == 'ST_GCN_Block':
                    i_stream.append(ST_GCN_Block(self.dim_in[0], self.stgcn_filters[0], 5, self.batch_size, self.cycle_size,
                                                self.num_nodes_per_graph, device, first=True))
                    for i in range(1, self.size_stgcn):
                        i_stream.append(ST_GCN_Block(self.stgcn_filters[i-1], self.stgcn_filters[i], 5, self.batch_size, self.cycle_size, self.num_nodes_per_graph, device))
                    self.streams.append(i_stream)
                elif self.model_type == 'Cust_STGCN_Block':
                    i_stream.append(Cust_STGCN_Block(self.dim_in[0], self.stgcn_filters[0], 5, self.batch_size, self.cycle_size,
                                                self.num_nodes_per_graph, device, first=True))
                    for i in range(1, self.size_stgcn):
                        i_stream.append(Cust_STGCN_Block(self.stgcn_filters[i-1], self.stgcn_filters[i], 5, self.batch_size, self.cycle_size, self.num_nodes_per_graph, device))
                    self.streams.append(i_stream)
                elif self.model_type == 'Gait_Graph2_Block':
                    i_stream.append(Gait_Graph2_Block(self.dim_in[0], self.stgcn_filters[0], 5, self.batch_size, self.cycle_size,
                                                self.num_nodes_per_graph, device, first=True))
                    for i in range(1, self.size_stgcn):
                        i_stream.append(Gait_Graph2_Block(self.stgcn_filters[i-1], self.stgcn_filters[i], 5, self.batch_size, self.cycle_size, self.num_nodes_per_graph, device))
                    self.streams.append(i_stream)
                elif self.model_type == 'STJA_GCN_Block':
                    i_stream.append(STJA_GCN_Block(self.dim_in[0], self.stgcn_filters[0], 5, self.batch_size, self.cycle_size,
                                                self.num_nodes_per_graph, device, first=True))
                    for i in range(1, self.size_stgcn):
                        i_stream.append(STJA_GCN_Block(self.stgcn_filters[i-1], self.stgcn_filters[i], 5, self.batch_size, self.cycle_size, self.num_nodes_per_graph, device))
                    self.streams.append(i_stream)
                elif self.model_type == 'ST_TAGCN_Block':
                    i_stream.append(ST_TAGCN_Block(self.dim_in[0], self.stgcn_filters[0], 5, self.batch_size, self.cycle_size,
                                                self.num_nodes_per_graph, device, first=True))
                    for i in range(1, self.size_stgcn):
                        i_stream.append(ST_TAGCN_Block(self.stgcn_filters[i-1], self.stgcn_filters[i], 5, self.batch_size, self.cycle_size, self.num_nodes_per_graph, device))
                    self.streams.append(i_stream)
                elif self.model_type == 'S_GCN':
                    i_stream.append(S_GCN(self.dim_in[0], self.stgcn_filters[0], 5, self.batch_size, self.cycle_size, self.num_nodes_per_graph,
                                                device, first=True))
                    for i in range(1, self.size_stgcn):
                        i_stream.append(S_GCN(self.stgcn_filters[i-1], self.stgcn_filters[i], 5, self.batch_size, self.cycle_size, self.num_nodes_per_graph, device, first=False))
                    self.streams.append(i_stream)
                elif self.model_type == 'T_GCN':
                    i_stream.append(T_GCN(self.dim_in[0], self.stgcn_filters[0], 5, self.batch_size, self.cycle_size, self.num_nodes_per_graph,
                                                device, first=True))
                    for i in range(1, self.size_stgcn):
                        i_stream.append(T_GCN(self.stgcn_filters[i-1], self.stgcn_filters[i], 5, self.batch_size, self.cycle_size, self.num_nodes_per_graph, device, first=False))
                    self.streams.append(i_stream)
                elif self.model_type == 'VAE':
                    i_stream.append(VAE_ST_TAGCN_Block(self.dim_in[0], self.stgcn_filters[0], 5, self.batch_size, self.cycle_size,
                                                device = device, first=True))
                    for i in range(1, self.size_stgcn):
                        i_stream.append(VAE_ST_TAGCN_Block(self.dim_in[0], self.stgcn_filters[0], 5, self.batch_size, self.cycle_size, device=device, first=False))
                    self.streams.append(i_stream)
                else:
                    print("entered type not valid")
        
        #Assign input value for final linear layers after combination
        self.len_steams = len(self.streams)
        if self.hcf:
            self.len_steams -= 1

        linear_input = self.stgcn_filters[-1] * self.cycle_size * self.len_steams
        print("Linear input info: ", self.stgcn_filters, dim_8th, len(self.streams), self.num_inputs, self.cycle_size, self.len_steams)

        #If HCF data is being used, append the length of it's final layer 
        #to the linear input. If there is only one stream and self.hcf is true,
        #then the HCF is the only data being processed.
        if self.hcf:
            if len(self.data_dims) != 1:
                linear_input += dim_8th
            else:
                linear_input = dim_8th
        
        linear_input = int(linear_input * 2)
        #print("Final: ", linear_input)

        self.avg_pool = nn.AvgPool2d(4, 4)
        self.combination_layer = torch.nn.Sequential(
        Linear(linear_input, 2048), ReLU(), BatchNorm1d(2048), torch.nn.Dropout(0.15),
        Linear(2048, 1024), ReLU(), BatchNorm1d(1024), torch.nn.Dropout(0.15),
        Linear(1024, 512), ReLU(), BatchNorm1d(512), torch.nn.Dropout(0.15),
        Linear(512, 128), ReLU(), BatchNorm1d(128), torch.nn.Dropout(0.15))
        #Linear(512, 18), ReLU(), BatchNorm1d(18), torch.nn.Dropout(0.15),
        #Linear(18, num_classes)
        #)
        self.second_last = torch.nn.Sequential(Linear(128, 18), ReLU(), BatchNorm1d(18), torch.nn.Dropout(0.15))
        self.last = torch.nn.Sequential(Linear(18, num_classes))

   
    def forward(self, data, edge_indices, batches, train):

        #Stage 1: Process the individual input streams
        hidden_layers = []
        for i, stream in enumerate(self.streams):
            #Get the data and pass it to the GPU
            data[i] = data[i].to(self.device)
            #If the data being passed is HCF data, it won't have edge indices.
            if edge_indices[i] != None:
                edge_indices[i] = edge_indices[i].to(self.device)
            batches[i] = batches[i].to(self.device)

            #Get the appropriate data for this stream
            h = data[i]
            #If this is an HCF layer, apply the first stream
            if self.hcf == True and len(self.streams) - 1 == i:
                #In the case of HCF this is a sequential Object
                h = stream(h)
                hidden_layers.append(h)
            else:
                #Reshape for temporal convolutions
                #batch, channel, num nodes per cycle, num features
                #print("h shape: ", h.shape, self.batch_size, self.cycle_size, self.dim_in, i)
                if self.model_type != 'VAE':
                    h = h.view(self.batch_size, self.dim_in[i], self.cycle_size)
                #In the case of ST-GCN this is a list object
                for j, layer in enumerate(stream):
                    if self.model_type == 'VAE':
                        h, var, mu, z = layer(h, edge_indices[i], train)
                    else:
                        h = layer(h, edge_indices[i], train)
                    #Add the last layer of each stream to a list. reshaping it to 2D first
                    #so it's compatible with HCF layers
                    #print("what's h: ", h.shape, type(h))
                    if j == len(stream) - 1:
                        if self.model_type != 'VAE':
                            #print("REMOVE THIS IN FUTURE")
                            h = h.view(h.shape[0], h.shape[2], h.shape[1])
                            h = h.view(h.shape[0], h.shape[1] * h.shape[2])
                        hidden_layers.append(h)

        # Concatenate graph embeddings
        #print("self.block", self.block2, type(self.block2))

        if self.model_type == 'VAE':
            h = torch.cat(([l for l in hidden_layers]), dim=0)
            #print("what am i returning at the end of graph network forward", h.shape, var.shape, mu.shape)
            return h, var, mu, z
        
        h = torch.cat(([l for l in hidden_layers]), dim=1)
        #h = h.view(h.shape[0], 14, -1)
        #print("size before: ", h.size())
        #h = self.avg_pool(h)
        #print("now: ", h.size())
        #h = h.view(h.shape[0], -1)
        # Combine the results and pass them through the combination layer
        #To compress them into classification
        #print("h in : ", h.size())
        h = self.combination_layer(h)
        embedding = self.second_last(h)
        h = self.last(embedding)
        #print("h out: ", h.size())
        return h, embedding