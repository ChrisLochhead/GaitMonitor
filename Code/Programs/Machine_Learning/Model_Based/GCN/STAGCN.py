import torch
torch.manual_seed(42)

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool,  GATv2Conv, ChebConv
from torch.nn import Linear, BatchNorm1d, ReLU
from Programs.Machine_Learning.Model_Based.GCN.Dataset_Obj import *
import torch.nn as nn

#Basic ST-GCN Block. These can only be stacked in lists not Torch.NN.Sequentials 
#because forward takes multiple inputs which causes problem even in custom sequential implementations.
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
        #print("in forward: ", x.shape)
        if self.batch_size > 1:
            x = self.b0(x)
        residual = x
        #print("skip connection and residual", self.skip_connection, residual.shape)
        residual = self.relu(self.skip_connection(residual))
        #Convert to 2D representation for GAT layer (Batch * Cycle, Channel)
        x = x.view(x.shape[2] * x.shape[0], x.shape[1])
        x = self.relu(self.b1(self.spatial_conv(x, edge_index)))
        x = self.relu(self.b2(self.spatial_conv_gat(x, edge_index)))

        #print("in forward 2: ", x.shape)
        #Convert to 3D representation for Temporal layer (Batch, Channel, Cycle)
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        x = torch.permute(x, (1, 0, 2))
        x = torch.transpose(x, 0, 1)
        x = torch.permute(x, (1, 0, 2))
        x = torch.transpose(x, 0, 1)

        x = residual + x
        x = self.dropout(x)
        return x
    
    #Basic ST-GCN Block. These can only be stacked in lists not Torch.NN.Sequentials 
#because forward takes multiple inputs which causes problem even in custom sequential implementations.
class STGCNBlock(torch.nn.Module):
    def __init__(self, in_channels, dim_h, temporal_kernel_size, batch_size, cycle_size, spatial_size, device, first = False):
        super(STGCNBlock, self).__init__()
        #Layers
        self.b0 = BatchNorm1d(in_channels).to(device)
        self.spatial_conv = ChebConv(in_channels, dim_h, 1).to("cuda")
        double_dim = int(dim_h * 2)

        self.b1 = BatchNorm1d(dim_h).to(device)
        self.b2 = BatchNorm1d(dim_h).to(device)
        self.temporal_conv2 = torch.nn.Conv1d(dim_h, int(dim_h), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)
        self.relu = ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.skip_connection = torch.nn.Conv1d(in_channels, int(dim_h), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)

        #Shape Info
        self.batch_size = batch_size
        self.cycle_size = cycle_size

    def forward(self, x, edge_index, train):
        #print("in forward: ", x.shape)
        if self.batch_size > 1:
            x = self.b0(x)

        residual = x
        #print("skip connection and residual", self.skip_connection, residual.shape)
        residual = self.relu(self.skip_connection(residual))
        #Convert to 2D representation for GAT layer (Batch * Cycle, Channel)
        x = x.view(x.shape[2] * x.shape[0], x.shape[1])
        x = self.relu(self.b1(self.spatial_conv(x, edge_index)))

        #Convert to 3D representation for Temporal layer (Batch, Channel, Cycle)
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        x = torch.permute(x, (1, 0, 2))
        x = torch.transpose(x, 0, 1)

        x = self.relu(self.b2(self.temporal_conv2(x)))
        x = torch.permute(x, (1, 0, 2))
        x = torch.transpose(x, 0, 1)

        x = residual + x
        x = self.dropout(x)
        return x
    

#Basic ST-GCN Block. These can only be stacked in lists not Torch.NN.Sequentials 
#because forward takes multiple inputs which causes problem even in custom sequential implementations.
class STAGCNBlock(torch.nn.Module):
    def __init__(self, in_channels, dim_h, temporal_kernel_size, batch_size, cycle_size, spatial_size, device, first = False):
        super(STAGCNBlock, self).__init__()
        #Layers
        if first:
            self.b0 = BatchNorm1d(in_channels).to(device) 
            self.spatial_conv = ChebConv(in_channels, int(dim_h*2), 1).to(device) 
            self.spatial_conv = GATv2Conv(in_channels, dim_h, heads=2).to(device)
            self.skip_connection = torch.nn.Conv1d(in_channels, int(dim_h), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)
            self.temp_att = GATv2Conv(int(dim_h*2), int(dim_h*2), heads=1).to(device)
        else:
            self.b0 = BatchNorm1d(int(in_channels * 2)).to(device)  
            self.spatial_conv = ChebConv(int(in_channels*2), int(dim_h*2), 1).to(device) 
            self.spatial_conv = GATv2Conv(int(in_channels*2), int(dim_h*2), heads=2).to(device) 
            self.skip_connection = torch.nn.Conv1d(int(in_channels*2), int(dim_h), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)           
            self.temp_att = GATv2Conv(int(dim_h*2), int(dim_h*2), heads=1).to(device)
        

        double_dim = int(dim_h * 2)

        self.b1 = BatchNorm1d(double_dim).to(device)
        self.b2 = BatchNorm1d(cycle_size).to(device)
        self.b3 = BatchNorm1d(int(dim_h)).to(device)
        self.temporal_conv2 = torch.nn.Conv1d(cycle_size, cycle_size, kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)
        self.relu = ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.in_channels = in_channels
        #Shape Info
        self.batch_size = batch_size
        self.cycle_size = cycle_size


    def forward(self, x, edge_index, train):
       # print("initial x: ", x.shape, self.in_channels)
        if self.batch_size > 1:
            x = self.b0(x)


        residual = x
        #print("skip connection and residual", self.skip_connection, residual.shape)
        residual = self.relu(self.skip_connection(residual))
        #print("after residual process", x.shape)

        #Convert to 2D representation for GAT layer (Batch * Cycle, Channel)
        x = x.view(x.shape[2] * x.shape[0], x.shape[1])
        #print("and here: ", x.shape)
        x = self.relu(self.b1(self.spatial_conv(x, edge_index)))

        #Convert to 3D representation for Temporal layer (Batch, Channel, Cycle)
       # print("herea: ", x.shape)
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        residual = x
        #print("hereb: ", x.shape)
        x = torch.permute(x, (0, 2, 1))

        #print("herec: ", x.shape)
        #Normal ST-GCN
        x = self.relu(self.b2(self.temporal_conv2(x)))
        #TEMPORAL ATTENTION ST-GCN
        #print("here1: ", x.shape)
        #x = x.view(x.shape[0] * x.shape[1], x.shape[2])
        #print("here2: ", x.shape, self.temp_att)
        #x = self.relu(self.b1(self.temp_att(x, edge_index)))
        #print("here3: ", x.shape)
        #x = x.view(self.batch_size, x.shape[1], self.cycle_size,)
        #print("here4: ", x.shape, residual.shape)
        #END OF TEMPORAL PART

       # print("hered: ", x.shape)
        #print("heree: ", x.shape, residual.shape, self.skip_connection)
        #THIS IS NEEDED WHEN USING NORMAL NON-TEMPORAL
        x = torch.permute(x, (0, 2, 1))

       #print("heref: ", x.shape, residual.shape, self.skip_connection)       
        x = residual + x
        x = self.dropout(x)
        #print("done", x.shape)
        return x

class GraphNetwork(torch.nn.Module):
    def __init__(self, dim_in, dim_h, num_classes, n_inputs, data_dims, batch_size, hcf = False, stgcn_size = 3, stgcn_filters = [64, 128, 256], 
                 max_cycle = 49, num_nodes_per_graph = 18, device = 'cuda', type = 0):
        super(GraphNetwork, self).__init__()

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
            #Every other possible input will be processed via an ST-GCN block which needs to be appended in a list instead of a sequential.
                i_stream = []
                if self.model_type == 0:
                    i_stream.append(GATBlock(self.dim_in[0], self.stgcn_filters[0], 5, self.batch_size, self.cycle_size,
                                                self.num_nodes_per_graph, device, first=True))
                    for i in range(1, self.size_stgcn):
                        i_stream.append(GATBlock(self.stgcn_filters[i-1], self.stgcn_filters[i], 5, self.batch_size, self.cycle_size, self.num_nodes_per_graph, device))
                elif self.model_type == 1:
                    i_stream.append(STGCNBlock(self.dim_in[0], self.stgcn_filters[0], 5, self.batch_size, self.cycle_size,
                                                self.num_nodes_per_graph, device, first=True))
                    for i in range(1, self.size_stgcn):
                        i_stream.append(STGCNBlock(self.stgcn_filters[i-1], self.stgcn_filters[i], 5, self.batch_size, self.cycle_size, self.num_nodes_per_graph, device))
                else:
                    i_stream.append(STAGCNBlock(self.dim_in[0], self.stgcn_filters[0], 5, self.batch_size, self.cycle_size,
                                                self.num_nodes_per_graph, device, first=True))
                    for i in range(1, self.size_stgcn):
                        i_stream.append(STAGCNBlock(self.stgcn_filters[i-1], self.stgcn_filters[i], 5, self.batch_size, self.cycle_size, self.num_nodes_per_graph, device))

                self.streams.append(i_stream)
        
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
        Linear(512, 128), ReLU(), BatchNorm1d(128), torch.nn.Dropout(0.15),
        Linear(128, num_classes)
        )

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
                h = h.view(self.batch_size, self.dim_in[i], self.cycle_size)
                #In the case of ST-GCN this is a list object
                for j, layer in enumerate(stream):
                    h = layer(h, edge_indices[i], train)
                    #Add the last layer of each stream to a list. reshaping it to 2D first
                    #so it's compatible with HCF layers
                    if j == len(stream) - 1:
                        h = h.view(h.shape[0], h.shape[1] * h.shape[2])
                        hidden_layers.append(h)

        # Concatenate graph embeddings
        h = torch.cat(([l for l in hidden_layers]), dim=1)
        #h = h.view(h.shape[0], 14, -1)
        #print("size before: ", h.size())
        #h = self.avg_pool(h)
        #print("now: ", h.size())
        #h = h.view(h.shape[0], -1)
        # Combine the results and pass them through the combination layer
        #To compress them into classification
        h = self.combination_layer(h)
        #print("h out: ", h.size())
        return h