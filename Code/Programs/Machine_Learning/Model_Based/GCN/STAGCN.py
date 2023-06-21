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
        #self.spatial_conv = ChebConv(dim_h, dim_h, K=3)
        self.b1 = BatchNorm1d(dim_h).to("cuda")
        self.temporal_conv2 = torch.nn.Conv1d(dim_h, dim_h, kernel_size=temporal_kernel_size, padding='same').to("cuda")
        self.relu = ReLU()
        self.pooling = torch.nn.MaxPool2d(kernel_size=(1, 0.5)).to("cuda")
        self.batch_size = batch_size
        self.cycle_size = cycle_size

    def forward(self, x, edge_index):
        #print("x", x.shape)
        x = self.relu(self.temporal_conv1(x))
        #Convert to 2D representation for GAT layer
        x = x.view(x.shape[2] * x.shape[0], x.shape[1])
        #print("before spatial convolution: ", x.shape)

        x = self.relu(self.spatial_conv(x, edge_index))
        #print("made it here 1", x.shape)
        x = self.b1(x)
        #Convert to 3D representation for Temporal layer
        x = x.view(self.batch_size, x.shape[1], self.cycle_size)
        x = self.relu(self.temporal_conv2(x))
        #print("made it to the end ", x.shape)
        return x

class MultiInputSTGACN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, num_classes, n_inputs, data_dims, batch_size, hcf = False, stgcn_size = 3, stgcn_filters = [64, 128, 256], 
                 multi_layer_feedback = False, cycle_size = 360):
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
        self.cycle_size = cycle_size

        for i in range(self.num_inputs):
            i_stream = []
            if i == len(range(self.num_inputs)) - 1 and hcf == True :
                 # This is a normal linear net for HCF data without graphical structure
                i_stream.append(Linear(dim_in[1], dim_h).to("cuda"))
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
        
        print("number of streams built: ", len(self.streams), self.streams[0][0], self.streams[1][0])

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

        self.fc1 = Linear(linear_input, 128)
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
        #print("original : ", x.shape)

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
                #print("being passed: ", 16, self.data_dims[stream_no][0], self.data_dims[stream_no][1])
                #print("shape before: ", h.shape)
                h = h.view(self.batch_size, self.dim_in, self.cycle_size)
                #print("shape now: ", h.shape)


            for i, layer in enumerate(stream):
                #This is the usual convolution block #if hcf, this is just a linear layer
                if stream_no == len(self.streams) - 1 and self.hcf:
                    #print("going in here?", i, stream_no, type(layer))
                    if i % 2 == 0:
                        h = F.relu(layer(h))
                        h = stream[i + 1](h)
                else:
                    #print("only in here")
                    #h = h.view(5760,9)
                    if i < self.size_stgcn:
                        #print("layer stgcn: ", layer, stream_no, type(layer), self.hcf, len(stream))
                        h = F.relu(layer(h, edge_indices[stream_no]))
                        h = F.dropout(h, p=0.1, training=train)

                #Record each  hidden layer value
                if self.hcf and stream_no + 1 == len(self.streams):
                    if i == len(stream) - 2:
                        #print("last layer appended only (hcf): ", i)
                        hidden_layer_stream.append(h)
                else:
                    #Only add the last layer (simplified)
                    if i == len(stream) - 1:
                        #print("appending layer in stream {} at layer {}".format(stream_no, i))
                        hidden_layer_stream.append(h)

            hidden_layers.append(hidden_layer_stream)
            

        #Concatenate all stream outputs
        h_layers = []
        for i, hidden_stream in enumerate(hidden_layers):
            #print("new stream", i)
            for j, layer in enumerate(hidden_stream):
                #print("stream {} layer {}".format(i, j))
                #print("size befoer: ", hidden_layers[i][j].shape, j)
                if i < len(hidden_layers) - 1:
                    hidden_layers[i][j] = hidden_layers[i][j].view(hidden_layers[i][j].shape[0] * hidden_layers[i][j].shape[2], hidden_layers[i][j].shape[1])
                #print("size: ", hidden_layers[i][j].shape, j)
                h_layers.append(global_add_pool(hidden_layers[i][j], batches[i]))

        # Concatenate graph embeddings
        h_out = torch.cat(([l for l in h_layers]), dim=1)


        #print("shape of concatenated output: ", h_out.shape)
        #print("convert the ST-GCN output to normal fc layer")
        #h = h.view(h.shape[0] * h.shape[2], h.shape[1])
        
        # Classifier
        h = F.relu(self.fc1(h_out))
        h = self.b1(h)
        h = F.dropout(h, p=0.1, training=train)
        h = F.relu(self.fc2(h))
        h = self.b2(h)
        h = F.dropout(h, p=0.1, training=train)
        h = self.fc3(h)

        return F.sigmoid(h), F.log_softmax(h, dim=1)