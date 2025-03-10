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
from Programs.Machine_Learning.GCN.Dataset_Obj import *
import torch.nn as nn
from torch_geometric.nn.conv import GraphConv, AGNNConv
from torchvision.models import resnet18

class GATBlock(torch.nn.Module):
    def __init__(self, in_channels, dim_h, temporal_kernel_size, batch_size, cycle_size, spatial_size, device, first = False):
        super(GATBlock, self).__init__()
        #Layers
        self.b0 = BatchNorm1d(in_channels).to(device)
        self.spatial_conv = GATv2Conv(in_channels, dim_h, heads=2).to(device)
        self.spatial_conv_gat = GATv2Conv(int(dim_h * 2), dim_h, heads=1).to(device)
        double_dim = int(dim_h * 2)
        self.b1 = BatchNorm1d(double_dim).to(device)
        self.b2 = BatchNorm1d(dim_h).to(device)
        self.temporal_conv2 = torch.nn.Conv1d(double_dim, int(double_dim/1), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)
        self.relu = ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.skip_connection = torch.nn.Conv1d(in_channels, int(double_dim/2), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)
        #Shape Info
        self.batch_size = batch_size
        self.cycle_size = cycle_size

    def forward(self, x, edge_index, train):
        if self.batch_size > 1:
            x = self.b0(x)
        residual = x
        residual = self.relu(self.skip_connection(residual))
        #Convert to 2D representation for GAT layer (Batch * Cycle, Channel)
        x = x.view(x.shape[2] * x.shape[0], x.shape[1])
        x = self.relu(self.b1(self.spatial_conv(x, edge_index)))
        x = self.relu(self.b2(self.spatial_conv_gat(x, edge_index)))
        #Convert to 3D representation for Temporal layer (Batch, Channel, Cycle)
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        x = torch.permute(x, (1, 0, 2))
        x = torch.transpose(x, 0, 1)
        x = torch.permute(x, (1, 0, 2))
        x = torch.transpose(x, 0, 1)
        x = residual + x
        x = self.dropout(x)
        return x
       
class Cust_STGCN_Block(torch.nn.Module):
    def __init__(self, in_channels, dim_h, temporal_kernel_size, batch_size, cycle_size, spatial_size, device, first = False):
        super(Cust_STGCN_Block, self).__init__()
        #Layers
        self.first = first
        if first:
            self.b0 = BatchNorm1d(in_channels).to(device) 
            self.spatial_conv = ChebConv(in_channels, int(dim_h * 2), 1).to(device)
            self.skip_connection = torch.nn.Conv1d(in_channels, int(dim_h*2), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)
        else:
            self.b0 = BatchNorm1d(int(in_channels * 2)).to(device)  
            self.spatial_conv = ChebConv(int(in_channels * 2 ), int(dim_h * 2), 1).to(device) 
            self.skip_connection = torch.nn.Conv1d(int(in_channels*2), int(dim_h*2), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)                     
        
        #custom ST-GCN
        self.spatial_conv_2 = ChebConv(int(dim_h*2), int(dim_h*2), 1).to(device)

        #regular st-gcn temporal
        self.temporal_conv = ChebConv(int(dim_h*2), int(dim_h*2), 1).to(device)
        double_dim = int(dim_h * 2)

        #gait graph comments
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
        #print("skip connection and residual", self.skip_connection, residual.shape, self.cycle_size, self.batch_size)
        residual = self.relu(self.skip_connection(residual))
        #print("after residual process", x.shape)
        #Convert to 2D representation for layer (Batch * Cycle, Channel)
        x = x.view(x.shape[2] * x.shape[0], x.shape[1])
        #Apply standard spatial convolution
        #print("spatial conv: ", self.spatial_conv)
        x = self.relu(self.b1(self.spatial_conv(x, edge_index)))
        #print("after spatial 1: ", x.shape)
        #only cust_st-gcn
        x = self.relu(self.b1(self.spatial_conv_2(x, edge_index)))   
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

class STJA_GCN_Block(torch.nn.Module):
    def __init__(self, in_channels, dim_h, temporal_kernel_size, batch_size, cycle_size, spatial_size, device, first = False):
        super(STJA_GCN_Block, self).__init__()
        #Layers
        self.first = first
        if first:
            self.b0 = BatchNorm1d(in_channels).to(device) 
            self.spatial_conv = GATv2Conv(in_channels, int(dim_h), heads=2).to(device)
            self.skip_connection = torch.nn.Conv1d(in_channels, int(dim_h*2), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)
        else:
            self.b0 = BatchNorm1d(int(in_channels * 2)).to(device)  
            self.spatial_conv = GATv2Conv(int(in_channels*2), int(dim_h*2), heads=1).to(device) 
            self.skip_connection = torch.nn.Conv1d(int(in_channels*2), int(dim_h*2), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)           
        


        self.temporal = ChebConv(int(dim_h*2), int(dim_h*2), 1).to(device)
        
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
        #print("skip connection and residual", self.skip_connection, residual.shape, self.cycle_size, self.batch_size)
        residual = self.relu(self.skip_connection(residual))
        #print("after residual process", x.shape)
        #Convert to 2D representation for GAT layer (Batch * Cycle, Channel)
        x = x.view(x.shape[2] * x.shape[0], x.shape[1])
        #Apply standard spatial convolution
        #print("spatial conv: ", self.spatial_conv)
        x = self.relu(self.b1(self.spatial_conv(x, edge_index)))
        #print("after spatial 1: ", x.shape)
        #Switch to temporal format (keep channels for the residual) maybe not needed
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        x = x.view(self.batch_size, self.cycle_size, -1)
        #Put it back into 2 dims
        x = x.view(x.shape[1] * x.shape[0], x.shape[2])
        #Convolve w/attention in the temporal direction
        x = self.relu(self.b1(self.temporal(x, edge_index)))
        #Restore to temporal shape
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        #Apply dropout and residual
        x = residual + x
        x = self.dropout(x)
        #print("layer done \n\n")
        return x
    
class ST_GCN_Block(torch.nn.Module):
    def __init__(self, in_channels, dim_h, temporal_kernel_size, batch_size, cycle_size, spatial_size, device, first = False):
        super(ST_GCN_Block, self).__init__()
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
        #print("skip connection and residual", self.skip_connection, residual.shape, self.cycle_size, self.batch_size)
        residual = self.relu(self.skip_connection(residual))
        #print("after residual process", x.shape)
        #Convert to 2D representation for GAT layer (Batch * Cycle, Channel)
        x = x.view(x.shape[2] * x.shape[0], x.shape[1])
        #Apply standard spatial convolution
        #print("spatial conv: ", self.spatial_conv)
        x = self.relu(self.b1(self.spatial_conv(x, edge_index)))
        #print("after spatial 1: ", x.shape)
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
        #print("done", x.shape)
        return x
    
class Gait_Graph2_Block(torch.nn.Module):
    def __init__(self, in_channels, dim_h, temporal_kernel_size, batch_size, cycle_size, spatial_size, device, first = False):
        super(Gait_Graph2_Block, self).__init__()
        #Layers
        self.first = first
        if first:
            self.b0 = BatchNorm1d(in_channels).to(device) 
            self.spatial_conv = GATv2Conv(in_channels, dim_h, heads=2).to(device)
            self.skip_connection = torch.nn.Conv1d(in_channels, int(dim_h*2), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)
        else:
            self.b0 = BatchNorm1d(int(in_channels * 2)).to(device)  
            self.spatial_conv = ChebConv(int(in_channels ), int(dim_h), 1).to(device) 
            self.skip_connection = torch.nn.Conv1d(int(in_channels*2), int(dim_h*2), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)           
            
        #for Gaitgraph2
        check = dim_h == int(in_channels *2)
        if check:
            dim_h = int(dim_h/2)

        self.spatial_bottleneck = ChebConv(int(in_channels*2), int(dim_h), 1).to(device)
        self.spatial_expand = ChebConv(int(int(dim_h)), int(dim_h*2), 1).to(device)            
        self.temporal_bottleneck = ChebConv(int(dim_h*2), int(int(dim_h)), 1).to(device)
        self.temporal_expand = ChebConv(int(dim_h), int(dim_h*2), 1).to(device)

        #regular st-gcn temporal
        if self.first:
            self.temporal_conv = ChebConv(int(dim_h*2), int(dim_h*2), 1).to(device)
        else:
            self.temporal_conv = ChebConv(int(dim_h), int(dim_h), 1).to(device)
   
        double_dim = int(dim_h * 2)
        #gait graph comments
        if first:
            self.b1 = BatchNorm1d(double_dim).to(device)
        else:
            if check:
                self.b1 = BatchNorm1d(int(dim_h*2)).to(device)
            else:
                self.b1 = BatchNorm1d(int(dim_h)).to(device)
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
        #print("skip connection and residual", self.skip_connection, residual.shape, self.cycle_size, self.batch_size)
        residual = self.relu(self.skip_connection(residual))
        #print("after residual process", x.shape)

        #Convert to 2D representation for GAT layer (Batch * Cycle, Channel)
        x = x.view(x.shape[2] * x.shape[0], x.shape[1])
        #only gaitgraph 2
        #print("before bottleneck 1: ", x.shape, self.spatial_bottleneck)
        if self.first == False:
           #print("bottle necking")
            x = self.spatial_bottleneck(x, edge_index)
        #print("after bottleneck 1: ", x.shape)
        #Apply standard spatial convolution
        #print("spatial conv: ", self.spatial_conv)
        x = self.relu(self.b1(self.spatial_conv(x, edge_index)))
        #print("after spatial 1: ", x.shape)
        #only gaitgraph2
        if self.first == False:
            #print("expanding")
            x = self.spatial_expand(x, edge_index)
        #print("after expand 1: ", x.shape)
        #print("x: ", residual.shape)
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        x = residual + x
        x = x.view(x.shape[2] * x.shape[0], x.shape[1])
        #Switch to temporal format (keep channels for the residual) maybe not needed
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        x = x.view(self.batch_size, self.cycle_size, -1)
        #ST-AAGCN W temp only
        #Put it back into 2 dims
        x = x.view(x.shape[1] * x.shape[0], x.shape[2])
        #only gaitgraph2
        #print("before bottleneck 2: ", x.shape)
        if self.first == False:
           x = self.temporal_bottleneck(x, edge_index)
        #print("after bottleneck 2: ", x.shape)
        #Convolve w/attention in the temporal direction
        x = self.relu(self.b1(self.temporal_conv(x, edge_index)))
        #print("before expand 2: ", x.shape)
        #only gaitgraph2
        if self.first == False:
            #print("expanding 2: ", x.shape)
            x = self.temporal_expand(x, edge_index)
        #print("after expand 2: ", x.shape)
        #Restore to temporal shape
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        #Apply dropout and residual
        x = residual + x
        x = self.dropout(x)
        #print("layer done \n\n")
        #print("done", x.shape)
        return x
    
class ST_TAGCN_Block(torch.nn.Module):
    def __init__(self, in_channels, dim_h, temporal_kernel_size, batch_size, cycle_size, spatial_size, device, first = False):
        super(ST_TAGCN_Block, self).__init__()
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
            
        self.temporal_conv = AGNNConv(int(dim_h*2), int(dim_h*2)).to(device)
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
        if self.batch_size > 1:
            x = self.b0(x)
        residual = x
        residual = self.relu(self.skip_connection(residual))
        #print("x coming in: ", x.shape)
        x = x.view(x.shape[2] * x.shape[0], x.shape[1])
        #print("x 1: ", x.shape)
        x = self.relu(self.b1(self.spatial_conv(x, edge_index)))
        #print("x 2: ", x.shape)
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        x = x.view(self.batch_size, self.cycle_size, -1)
        x = x.view(x.shape[1] * x.shape[0], x.shape[2])
        #print("x 3: ", x.shape)
        x = self.relu(self.b1(self.temporal_conv(x, edge_index)))
        #print("x 4: ", x.shape)
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        #print("x out: ", x.shape)
        x = residual + x
        x = self.dropout(x)
        return x

class VAE_ST_TAGCN_Block(torch.nn.Module):
    def __init__(self, in_channels, dim_h, temporal_kernel_size, batch_size, cycle_size, device, first=False, latent_dim = 20):
        super(VAE_ST_TAGCN_Block, self).__init__()
        
        # Encoder layers
        self.encoder = Encoder(in_channels, dim_h, temporal_kernel_size, batch_size, cycle_size, latent_dim, device, first)
        
        # Decoder layers
        self.decoder = Decoder(in_channels, dim_h, temporal_kernel_size, batch_size, cycle_size, device)
        
    def forward(self, x, edge_index, train=True):
        # Encode input data to latent space

        x, mu, log_var = self.encoder(x, edge_index)
    
        # Reparameterization trick
        std = log_var.mul(0.5).exp_()
        eps = torch.randn_like(log_var)
        z = eps.mul(std).add_(mu)
        
        # Decode latent space representation
        reconstructed_x = self.decoder(z, edge_index)
        #print("out here: ", reconstructed_x.shape, first.shape, x.shape)
        #stop = 5/0
        #return x, mu, log_var
        return reconstructed_x, mu, log_var, z #reconstructed_x
    

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, dim_h, temporal_kernel_size, batch_size, cycle_size, latent_dim, device, first=False):
        super(Encoder, self).__init__()
        # Layers
        self.first = first
        self.b0 = BatchNorm1d(in_channels).to(device) 
        self.start_shape = int(in_channels * cycle_size)
        #in_channels = self.start_shape
        print("shape going in: ", cycle_size, in_channels, batch_size)
        cycle_size  = cycle_size * in_channels #* batch_size
        print("cycle size: ", cycle_size)
        self.c1 = nn.Linear(cycle_size, int(cycle_size * 0.99)).to(device)
        self.c2 = nn.Linear(int(cycle_size * 0.99), int(cycle_size * 0.98)).to(device)
        self.c3 = nn.Linear(int(cycle_size), int(cycle_size * 0.8)).to(device)

        self.b1 = BatchNorm1d(int(cycle_size * 0.90)).to(device)
        self.b2 = BatchNorm1d(int(cycle_size * 0.85)).to(device)
        self.b3 = BatchNorm1d(int(cycle_size * 0.80)).to(device)

        self.relu = ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.cycle_size = cycle_size
        self.in_channels =self.start_shape# in_channels
        self.batch_size = batch_size
        # Shape Info
        self.latent_dim = int(self.cycle_size/4)
        self.mean = nn.Linear(int(cycle_size * 0.8), int(cycle_size * 0.8)).to(device)        
        self.var = nn.Linear(int(cycle_size * 0.8), int(cycle_size * 0.8)).to(device) 

    def forward(self, x, edge_index, train=True):
        #if self.batch_size > 1:
        ##    x = self.b0(x)
        x = x.view(self.batch_size, -1)
        #x = torch.flatten(x)
        #print("entering encoder: ", x.shape)
        #print("what does it look like going in: ", x)
        #residual = x
        ##residual = self.relu(self.skip_connection(residual))
        #print("starting shape: ", x.shape, self.cycle_size, self.batch_size, self.in_channels)
        #x = x.view(x.shape[0], x.shape[2], -1)
        #print("before spatial: ", x.shape, self.c3)
        ##x = self.relu(self.c1(x))
        #x = self.relu(self.c2(x))
        x = self.c3(x)
        #print("aFTER spatial: ", x.shape)
        #stop = 5/0
        #x = x.view(x.shape[0], x.shape[2], x.shape[1])
        #print("aFTER reshape: ", x.shape, self.temporal_conv, self.b2)
        #x = self.relu(self.b2(self.temporal_conv(x)))
        #print("aFTER temporal: ", x.shape)
        #x = x.view(x.shape[0], x.shape[2], x.shape[1])
        #x = x.view(x.shape[0] * x.shape[1], x.shape[2])
        #x = residual + x
        #x = self.dropout(x)
        
        # Latent space representation
        #print("aFTER dropout: ", x.shape, self.latent_dim)
        mu = self.mean(x)
        log_var = self.var(x)

        #print("final out of encoder: ", x, x.shape)
        #print("\n")
        return x, mu, log_var
    

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, dim_h, temporal_kernel_size, batch_size, cycle_size, device):
        super(Decoder, self).__init__()
        self.cycle_size = cycle_size
        cycle_size  = 3 * cycle_size
        self.c1 = nn.Linear(int(cycle_size * 0.97), int(cycle_size * 0.98)).to(device)
        self.c2 = nn.Linear(int(cycle_size * 0.98), int(cycle_size * 0.99)).to(device)
        self.c3 = nn.Linear(int(cycle_size * 0.8), int(cycle_size)).to(device)
        self.c4 = nn.Linear(int(cycle_size * 0.95), cycle_size).to(device)
        self.b1 = BatchNorm1d(106).to(device)
        self.b2 = BatchNorm1d(int(cycle_size * 0.66)).to(device)
        self.b3 = BatchNorm1d(int(cycle_size)).to(device)

        self.relu = ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        # Shape Info
        self.batch_size = batch_size

    def forward(self, x, edge_index, train=True):
        #print("input sizes: ", x.shape)
        #x = x.view(x.shape[0], x.shape[2], x.shape[1])
        #print("before temp:", x.shape)
        #x = self.relu(self.b1(self.temporal_conv(x)))
        #x = x.view(x.shape[0], x.shape[2], x.shape[1])
       # x = self.relu(self.c1(x))
        #print("after c1:", x.shape, self.c1)
        #x = self.relu(self.c2(x))
        #print("after c2:", x.shape, self.c2)
        x = self.c3(x)
        #print("after c3:", x.shape, self.c3)
        #x = self.c4(x)
        #print("after spatial:", x.shape)
        #x = x.view(x.shape[0], x.shape[2], x.shape[1])
        #x = self.dropout(x)
        #x = x.view(x.shape[1], x.shape[0])
        x = x.view(self.batch_size, -1)
        x = torch.sigmoid(x)
        x = x.view(self.batch_size * self.cycle_size, -1)
        #print("leaving decoder: ", x.shape)
        #print("what it looks like: ", x)
        #stop = 5/0
        return x