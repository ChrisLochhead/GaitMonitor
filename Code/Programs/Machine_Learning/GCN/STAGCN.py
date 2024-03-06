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
        self.first = first
        if first:
            self.b0 = BatchNorm1d(in_channels).to(device) 
            #self.spatial_conv = GraphConv(in_channels, int(dim_h*2)).to(device) 
            self.spatial_conv = GATv2Conv(in_channels, dim_h, heads=2).to(device)
            #self.spatial_conv = ChebConv(in_channels, int(dim_h*2), 1).to(device)
            self.skip_connection = torch.nn.Conv1d(in_channels, int(dim_h*2), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)
            #self.temp_att = GATv2Conv(int(dim_h*2), int(dim_h*2), heads=1).to(device)
        else:
            self.b0 = BatchNorm1d(int(in_channels * 2)).to(device)  
            #no attention, remove * 2 for gait graph
            self.spatial_conv = ChebConv(int(in_channels ), int(dim_h), 1).to(device) 
            #spatial attention
            #self.spatial_conv = GATv2Conv(int(in_channels*2), int(dim_h*2), heads=1).to(device) 

            self.skip_connection = torch.nn.Conv1d(int(in_channels*2), int(dim_h*2), kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)           
        

        #For ablation studies
        #self.temp_att = GATv2Conv(int(dim_h*2), int(dim_h*2), heads=1).to(device)
        #self.temp_att2 = GATv2Conv(int(dim_h*2), int(dim_h*2), heads=1).to(device)
            
        #for Gaitgraph2
        check = dim_h == int(in_channels *2)
        if check:
            dim_h = int(dim_h/2)

        self.spatial_bottleneck = ChebConv(int(in_channels*2), int(dim_h), 1).to(device)
        self.spatial_expand = ChebConv(int(int(dim_h)), int(dim_h*2), 1).to(device)            
        self.temporal_bottleneck = ChebConv(int(dim_h*2), int(int(dim_h)), 1).to(device)
        self.temporal_expand = ChebConv(int(dim_h), int(dim_h*2), 1).to(device)
              
        #custom ST-GCN
        #self.spatial_conv_2 = ChebConv(int(dim_h*2), int(dim_h*2), 1).to(device)

        #regular st-gcn temporal
        if self.first:
            self.temp_att = ChebConv(int(dim_h*2), int(dim_h*2), 1).to(device)
            self.temp_att2 = ChebConv(int(dim_h*2), int(dim_h*2), 1).to(device)
        else:
            self.temp_att = ChebConv(int(dim_h), int(dim_h), 1).to(device)
            self.temp_att2 = ChebConv(int(dim_h), int(dim_h), 1).to(device)           
        
        #ST-TAGCN
        #self.temp_att = AGNNConv(int(dim_h*2), int(dim_h*2)).to(device)
        #self.temp_att2 = AGNNConv(int(dim_h*2), int(dim_h*2)).to(device)

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
        #Used only in st-aagcn, this compensates for the removal of a normal layer to be replaced with the attention layer on the conv layer
        self.other_spatial_conv = ChebConv(cycle_size, cycle_size, 1).to(device) 
        self.temporal_conv2 = torch.nn.Conv1d(cycle_size, cycle_size, kernel_size=temporal_kernel_size, stride=1, padding='same').to(device)
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

        #only cust_st-gcn
        #x = self.relu(self.b1(self.spatial_conv_2(x, edge_index)))   

        #only gaitgraph2
        if self.first == False:
            #print("expanding")
            x = self.spatial_expand(x, edge_index)
        #print("after expand 1: ", x.shape)
        #print("x: ", residual.shape)
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        x = residual + x
        x = x.view(x.shape[2] * x.shape[0], x.shape[1])

        #ST-AAGCN only: apply spatial attention
        #x = self.relu(self.b1(self.temp_att(x, edge_index)))

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
        x = self.relu(self.b1(self.temp_att2(x, edge_index)))

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

class GraphNetwork(torch.nn.Module):
    def __init__(self, dim_in, dim_h, num_classes, n_inputs, data_dims, batch_size, hcf = False, stgcn_size = 3, stgcn_filters = [128, 128, 128, 128], 
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

        #Comparison network: basic DNN
        self.basic_dnn = torch.nn.Sequential(
        nn.Linear(324, 512), ReLU(), nn.BatchNorm1d(512), torch.nn.Dropout(0.15),
        nn.Linear(512, 256), ReLU(), nn.BatchNorm1d(256), torch.nn.Dropout(0.15),
        nn.Linear(256, 128), ReLU(), nn.BatchNorm1d(128), torch.nn.Dropout(0.15),
        nn.Linear(128,  num_classes)
        )

        self.resnet_out = torch.nn.Sequential(
        nn.Linear(13824, 1024), ReLU(), nn.BatchNorm1d(1024), torch.nn.Dropout(0.15),
        nn.Linear(1024, 128), ReLU(), nn.BatchNorm1d(128), torch.nn.Dropout(0.15),
        nn.Linear(128, num_classes)
        )
        #Resnet comparison
        self.resnet = resnet18(pretrained=False)

        #C3D Model implementation
        self.base_channels = self.stgcn_filters[0]
        k_size = (2, 2, 1)
        k_stride = (2, 2, 1)
        self.conv1a = nn.Conv3d(self.dim_in[0], self.base_channels, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = nn.Conv3d(self.base_channels, self.base_channels * 2, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool3d(kernel_size=k_size, stride=k_stride)

        self.conv3a = nn.Conv3d(self.base_channels * 2, self.base_channels * 4, kernel_size=3, padding=1)
        self.conv3b = nn.Conv3d(self.base_channels * 4, self.base_channels * 4, kernel_size=3, padding=1)
        self.conv3b_skip = nn.Conv3d(self.base_channels, self.base_channels * 4, kernel_size=3, padding=1)
        self.pool3 = nn.AvgPool3d(kernel_size=k_size, stride=k_stride)

        self.conv4a = nn.Conv3d(self.base_channels * 4, self.base_channels * 8, kernel_size=3, padding=1)
        self.conv4b = nn.Conv3d(self.base_channels * 8, self.base_channels * 8, kernel_size=3, padding=1)
        self.conv4b_skip = nn.Conv3d(self.base_channels, self.base_channels * 8, kernel_size=3, padding=1)

        self.pool4 = nn.AvgPool3d(kernel_size=k_size, stride=k_stride)
        self.conv5a = nn.Conv3d(self.base_channels * 8, self.base_channels * 8, kernel_size=3, padding=1)
        self.conv5b = nn.Conv3d(self.base_channels * 8, self.base_channels * 8, kernel_size=3, padding=1)

        #self.c3d_out = torch.nn.Sequential(
        #self.conv1a, self.pool1, self.conv2a, self.pool2, self.conv3a, self.conv3b,
        #self.pool3, self.conv4a, self.conv4b, self.pool4, self.conv5a, self.conv5b
        #)

        self.c3d_out1 = torch.nn.Sequential(
        self.conv1a, self.pool1, self.conv2a
        )


        self.c3d_out2 = torch.nn.Sequential(
        self.conv3a, self.conv3b,
        )

        self.c3d_out3 = torch.nn.Sequential(
         self.conv4a, self.conv4b)

        self.c3d_out4 = torch.nn.Sequential(
        self.conv5a, self.conv5b, torch.nn.Dropout(0.5)
        )
    '''
    #C3D Forward
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
            #print("h shape: ", h.shape)
            h = h.view(-1, self.batch_size, self.cycle_size)
            #print("h shape: ", h.shape)
            h = h.view(64, 3, int(self.cycle_size/6), 6)
            #print("h shape: ", h.shape)
            #h = self.basic_dnn(h)
            #h = self.c3d_out(h)
            res1 = self.conv3b_skip(h)
            res2 = self.conv3b_skip(h)

            h = h.view(3, 64, int(self.cycle_size/6), 6)
            h = self.c3d_out1(h)
            h = self.c3d_out2(h)
            h +=res1
            h = self.c3d_out3(h)
            h += res2
            h = self.c3d_out4(h)
            h += res2
            #print("out shape: ", h.shape)
            h = h.view(self.batch_size, 64, -1)
            #print("in shape: ", h.shape)
            h = h.view(h.shape[0], h.shape[1] * h.shape[2])
            #print("in shape: ", h.shape)
            h= self.resnet_out(h)
            break
        #print("h out: ", h.size())
        return h
    
    
    #DNN forward
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
            #print("h shape: ", h.shape)
            h = h.view(-1, self.batch_size, self.cycle_size)
            #print("h shape: ", h.shape)
            h = h.view(64, 3, int(self.cycle_size/6), 6)
            #print("h shape: ", h.shape)
            #h = self.basic_dnn(h)
            h = self.resnet(h)
            h= self.resnet_out(h)
            break
        #print("h out: ", h.size())
        return h
    '''
    #Normal forward
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
        #print("self.block", self.block2, type(self.block2))
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
        