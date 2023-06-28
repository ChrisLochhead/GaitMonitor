import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool,  GATv2Conv
from torch.nn import Linear, BatchNorm1d, ReLU
from Programs.Machine_Learning.Model_Based.GCN.Dataset_Obj import *
from torch_geometric.nn import ChebConv

class STGCNBlock(torch.nn.Module):
    def __init__(self, in_channels, dim_h, temporal_kernel_size, batch_size, cycle_size):
        super(STGCNBlock, self).__init__()
        self.temporal_conv1 = torch.nn.Conv1d(in_channels, dim_h, kernel_size=temporal_kernel_size, padding='same').to("cuda")
        self.spatial_conv = GATv2Conv(dim_h, dim_h, heads=1).to("cuda")
 
        self.b1 = BatchNorm1d(dim_h).to("cuda")
        self.temporal_conv2 = torch.nn.Conv1d(dim_h, dim_h, kernel_size=temporal_kernel_size, padding='same').to("cuda")
        self.relu = ReLU()
        self.pooling = torch.nn.MaxPool2d(kernel_size=(1, 0.5)).to("cuda")
        self.batch_size = batch_size
        self.cycle_size = cycle_size

    def forward(self, x, edge_index):
        x = self.relu(self.temporal_conv1(x))
        #Convert to 2D representation for GAT layer (Batch * Cycle, Channel)
        x = x.view(x.shape[2] * x.shape[0], x.shape[1])
        x = self.relu(self.spatial_conv(x, edge_index))
        x = self.b1(x)
        #Convert to 3D representation for Temporal layer (Batch, Channel, Cycle)
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        x = self.relu(self.temporal_conv2(x))
        return x

class MultiInputSTGACN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, num_classes, n_inputs, data_dims, batch_size, hcf = False, stgcn_size = 2, stgcn_filters = [32, 64], 
                 multi_layer_feedback = False, max_cycle = 20, num_nodes_per_graph = 18):
        super(MultiInputSTGACN, self).__init__()

        dim_half = int(dim_h/2)
        dim_4th = int(dim_half/2)
        dim_8th = int(dim_4th/2)
        self.dim_in = dim_in[0]
        self.num_inputs = n_inputs
        self.streams = []
        self.hcf = hcf
        self.size_stgcn = stgcn_size
        self.stgcn_filters = stgcn_filters
        self.batch_size = batch_size
        self.data_dims = data_dims
        self.num_nodes_per_graph = num_nodes_per_graph
        self.cycle_size = max_cycle * num_nodes_per_graph

        for i in range(self.num_inputs):
            i_stream = []
            if i == len(range(self.num_inputs)) - 1 and hcf == True :
                 # This is a normal linear net for HCF data without graphical structure
                i_stream.append(Linear(dim_in[-1], dim_h).to("cuda"))
                i_stream.append(BatchNorm1d(dim_h).to("cuda"))
                i_stream.append(Linear(dim_h, dim_half).to("cuda"))
                i_stream.append(BatchNorm1d(dim_half).to("cuda"))
                i_stream.append(Linear(dim_half, dim_4th).to("cuda"))
                i_stream.append(BatchNorm1d(dim_4th).to("cuda"))
                i_stream.append(Linear(dim_4th, dim_8th).to("cuda"))
                i_stream.append(BatchNorm1d(dim_8th).to("cuda"))
            else:
                i_stream.append(STGCNBlock(dim_in[0], self.stgcn_filters[0], 5, self.batch_size, self.cycle_size))
                for i in range(self.size_stgcn):
                    if i > 0:
                        i_stream.append(STGCNBlock(self.stgcn_filters[i-1], self.stgcn_filters[i], 5, self.batch_size, self.cycle_size))

            self.streams.append(i_stream)
        
        #print("number of streams built: ", len(self.streams), self.streams[0][0], self.streams[1][0])

        #Final processing after combination
         #Extra linear layer to compensate for more data

        if multi_layer_feedback:
            linear_input = dim_h * 2 * (len(self.data_dims) -1)
            #HCF only concatenates the last (or smallest) hidden layer, GAT convs take all 4 layers
        else:
            linear_input = self.stgcn_filters[-1]

        if self.hcf:
            #Add output of final hcf layer
            linear_input += dim_8th
            
        if self.hcf and len(self.data_dims) == 1:
            linear_input = dim_8th

        self.fc1 = Linear(self.cycle_size * self.stgcn_filters[-1], 128)
        self.b1 = BatchNorm1d(128)
        self.fc2 = Linear(128, 64)
        self.b2 = BatchNorm1d(64)
        self.fc3 = Linear(64, num_classes)

    def forward(self, data, edge_indices, batches, train):
        for i, x in enumerate(data):
            data[i] = data[i].to("cuda")
            if edge_indices[i] != None:
                edge_indices[i] = edge_indices[i].to("cuda")
            batches[i] = batches[i].to("cuda")

        hidden_layers = []
    
        for stream_no, stream in enumerate(self.streams):
            hidden_layer_stream = []
            h = data[stream_no]
            if self.hcf == True and len(self.streams) - 1 == stream_no:
                h = data[stream_no]
            else:
                #Reshape for temporal convolutions
                h = data[stream_no]
                #batch, channel, num nodes per cycle, num features
                h = h.view(self.batch_size, self.dim_in, self.cycle_size)

            for i, layer in enumerate(stream):
                #This is the usual convolution block #if hcf, this is just a linear layer
                if stream_no == len(self.streams) - 1 and self.hcf:
                    if i % 2 == 0:
                        h = F.relu(layer(h))
                        h = stream[i + 1](h)
                else:

                    if i < self.size_stgcn:
                        h = F.relu(layer(h, edge_indices[stream_no]))
                        h = F.dropout(h, p=0.1, training=train)

                #Record each  hidden layer value
                if self.hcf and stream_no + 1 == len(self.streams):
                    if i == len(stream) - 2:
                        hidden_layer_stream.append(h)
                else:
                    #Only add the last layer (simplified)
                    if i == len(stream) - 1:
                        hidden_layer_stream.append(h)

            hidden_layers.append(hidden_layer_stream)
            

        # Concatenate graph embeddings
        h = torch.cat(([l[-1] for l in hidden_layers]), dim=1)

        # Classifier
        h = h.view(h.shape[0], h.shape[1] * h.shape[2])
        h = F.relu(self.fc1(h))

        h = self.b1(h)
        h = F.dropout(h, p=0.1, training=train)
        h = F.relu(self.fc2(h))
        h = self.b2(h)
        h = self.fc3(h)
        print("finished?")

        return F.sigmoid(h), F.log_softmax(h, dim=1)

class STGACN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, num_classes, n_inputs, data_dims, batch_size, stgcn_size = 2, stgcn_filters = [32, 64], 
                 multi_layer_feedback = False, max_cycle = 20, num_nodes_per_graph = 18):
        super(STGACN, self).__init__()

        dim_half = int(dim_h/2)
        dim_4th = int(dim_half/2)
        dim_8th = int(dim_4th/2)
        self.dim_in = dim_in
        self.num_inputs = n_inputs
        self.stream = []
        self.size_stgcn = stgcn_size
        self.stgcn_filters = stgcn_filters
        self.batch_size = batch_size
        self.data_dims = data_dims
        self.num_nodes_per_graph = num_nodes_per_graph
        self.cycle_size = max_cycle * num_nodes_per_graph


        self.stream.append(STGCNBlock(dim_in, self.stgcn_filters[0], 5, self.batch_size, self.cycle_size))
        for i in range(self.size_stgcn):
            if i > 0:
                self.stream.append(STGCNBlock(self.stgcn_filters[i-1], self.stgcn_filters[i], 5, self.batch_size, self.cycle_size))


        self.fc1 = Linear(self.cycle_size * self.stgcn_filters[-1], 128)
        self.b1 = BatchNorm1d(128)
        self.fc2 = Linear(128, 64)
        self.b2 = BatchNorm1d(64)
        self.fc3 = Linear(64, num_classes)

    def forward(self, data, edge_indices, batches, train):
        for i, x in enumerate(data):
            data[i] = data[i].to("cuda")
            edge_indices[i] = edge_indices[i].to("cuda")
            batches[i] = batches[i].to("cuda")
  
            #Reshape for temporal convolutions
            h = data[i]
            #batch, channel, num nodes per cycle, num features
            h = h.view(self.batch_size, self.dim_in, self.cycle_size)

            for j, layer in enumerate(self.stream):
                #This is the usual convolution block #if hcf, this is just a linear layer
                if j < self.size_stgcn - 1:
                    h = F.relu(layer(h, edge_indices[i]))
                    h = F.dropout(h, p=0.5, training=train)
                else:
                    h = layer(h, edge_indices[i])

            # Classifier
            h = h.view(h.shape[0], h.shape[1] * h.shape[2])
            h = F.relu(self.fc1(h))

            h = self.b1(h)
            h = F.dropout(h, p=0.1, training=train)
            h = F.relu(self.fc2(h))
            h = self.b2(h)
            h = self.fc3(h)

        return F.sigmoid(h), h# F.log_softmax(h, dim=1)