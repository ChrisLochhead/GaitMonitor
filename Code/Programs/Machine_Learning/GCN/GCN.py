'''
GCN implementation
'''
#imports
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
#dependencies
from Programs.Machine_Learning.GCN.Dataset_Obj import *

class GCN(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, dim_in, dim_h, dim_out):
      super().__init__()
      self.gcn1 = GCNConv(dim_in, dim_h)
      self.gcn2 = GCNConv(dim_h, dim_h)
      self.lin1 = Linear(dim_h * 2, dim_h * 2)
      self.lin2 = Linear(dim_h * 2, dim_out)
      self.optimizer = torch.optim.Adam(self.parameters(),
                                        lr=0.01,
                                        weight_decay=5e-4)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        x = x.to("cuda")
        edge_index = edge_index.to("cuda")
        batch = batch.to("cuda")
        h1 = F.dropout(x, p=0.6, training=self.training)
        h1 = F.relu(self.gcn1(h1, edge_index))
        h2 = F.dropout(h1, p=0.6, training=self.training)
        h2 = F.relu(self.gcn2(h2, edge_index))

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        # Concatenate graph embeddings
        h = torch.cat((h1, h2), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h, F.log_softmax(h, dim=1)

from torch_geometric.nn import global_add_pool,  GATv2Conv, ChebConv
from torch.nn import Linear, BatchNorm1d, ReLU

class S_GCN(torch.nn.Module):
    def __init__(self, in_channels, dim_h, temporal_kernel_size, batch_size, cycle_size, spatial_size, device, first = False):
        super(S_GCN, self).__init__()
        #Layers
        self.first = first
        if first:
            self.b0 = BatchNorm1d(in_channels).to(device) 
            self.spatial_conv = ChebConv(in_channels, int(dim_h*2), 1).to(device)
            self.skip_connection = torch.nn.Conv1d(in_channels, int(dim_h*2), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)
        else:
            self.b0 = BatchNorm1d(int(in_channels * 2)).to(device)  
            self.spatial_conv = ChebConv(int(in_channels * 2 ), int(dim_h * 2), 1).to(device) 
            self.skip_connection = torch.nn.Conv1d(int(in_channels*2), int(dim_h*2), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)           
            

        double_dim = int(dim_h * 2)
        self.b1 = BatchNorm1d(double_dim).to(device)
        self.b2 = BatchNorm1d(cycle_size).to(device)
        self.b3 = BatchNorm1d(int(dim_h)).to(device)
        self.relu = ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.in_channels = in_channels
        #Shape Info
        self.batch_size = batch_size
        self.cycle_size = cycle_size

    def forward(self, x, edge_index, train):
        #print("initial x: ", x.shape, self.in_channels)
        if self.batch_size > 1:
            x = self.b0(x)

        residual = x
        residual = self.relu(self.skip_connection(residual))
        #Convert to 2D representation for GAT layer (Batch * Cycle, Channel)
        x = x.view(x.shape[2] * x.shape[0], x.shape[1])
        x = self.relu(self.b1(self.spatial_conv(x, edge_index)))
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        x = x.view(self.batch_size, self.cycle_size, -1)
        x = x.view(x.shape[1] * x.shape[0], x.shape[2])
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        #Apply dropout and residual
        x = residual + x
        x = self.dropout(x)
        return x
    

class T_GCN(torch.nn.Module):
    def __init__(self, in_channels, dim_h, temporal_kernel_size, batch_size, cycle_size, spatial_size, device, first = False):
        super(T_GCN, self).__init__()
        #Layers
        self.first = first
        if first:
            self.b0 = BatchNorm1d(in_channels).to(device) 
            self.spatial_conv = ChebConv(in_channels, int(dim_h*2), 1).to(device)
            self.temporal_conv = ChebConv(in_channels, int(dim_h*2), 1).to(device)
            self.skip_connection = torch.nn.Conv1d(in_channels, int(dim_h*2), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)
        else:
            self.b0 = BatchNorm1d(int(in_channels * 2)).to(device)  
            self.spatial_conv = ChebConv(int(in_channels * 2 ), int(dim_h * 2), 1).to(device) 
            self.skip_connection = torch.nn.Conv1d(int(in_channels*2), int(dim_h*2), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)           
            
            self.temporal_conv = ChebConv(int(dim_h*2), int(dim_h*2), 1).to(device)

        double_dim = int(dim_h * 2)
        self.b1 = BatchNorm1d(double_dim).to(device)
        self.b2 = BatchNorm1d(cycle_size).to(device)
        self.b3 = BatchNorm1d(int(dim_h)).to(device)
        self.relu = ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.in_channels = in_channels
        #Shape Info
        self.batch_size = batch_size
        self.cycle_size = cycle_size

    def forward(self, x, edge_index, train):
        #print("initial x: ", x.shape, self.in_channels)
        if self.batch_size > 1:
            x = self.b0(x)

        residual = x
        residual = self.relu(self.skip_connection(residual))
        #print("after residual process", x.shape)
        #Convert to 2D representation for GAT layer (Batch * Cycle, Channel)
        x = x.view(x.shape[2] * x.shape[0], x.shape[1])
        #Apply standard spatial convolution
        #x = self.relu(self.b1(self.spatial_conv(x, edge_index)))
        #Switch to temporal format (keep channels for the residual) maybe not needed
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        x = x.view(self.batch_size, self.cycle_size, -1)
        #Put it back into 2 dims
        x = x.view(x.shape[1] * x.shape[0], x.shape[2])
        #Convolve w/attention in the temporal direction
        x = self.relu(self.b1(self.temporal_conv(x, edge_index)))
        #Restore to temporal shape
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        #Apply dropout and residual
        x = residual + x
        x = self.dropout(x)
        #print("layer done \n\n")
        return x