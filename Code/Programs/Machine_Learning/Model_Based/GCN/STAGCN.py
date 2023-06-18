import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool,  GATv2Conv
from torch.nn import Linear, BatchNorm1d, ReLU
from Programs.Machine_Learning.Model_Based.GCN.Dataset_Obj import *
from torch_geometric.nn import ChebConv

class STGCNBlock(torch.nn.Module):
    def __init__(self, in_channels, dim_h, temporal_kernel_size):
        super(STGCNBlock, self).__init__()
        self.temporal_conv1 = torch.nn.Conv2d(in_channels, dim_h, kernel_size=(temporal_kernel_size, 1))
        self.spatial_conv = GATv2Conv(dim_h, dim_h, heads=1)
        #self.spatial_conv = ChebConv(dim_h, dim_h, K=3)
        self.b1 = BatchNorm1d(dim_h)
        self.temporal_conv2 = torch.nn.Conv2d(dim_h, in_channels, kernel_size=(temporal_kernel_size, 1))
        self.relu = ReLU()
        self.pooling = torch.nn.MaxPool2d(kernel_size=(1, 0.5))

    def forward(self, x, edge_index):
        x = self.relu(self.temporal_conv1(x))
        x = self.relu(self.spatial_conv(x, edge_index))
        x = self.b1(x)
        x = self.relu(self.temporal_conv2(x))
        return x

class STAGCN(torch.nn.Module):
    def __init__(self, num_nodes, num_features, dim_h, num_classes):
        super(STAGCN, self).__init__()
        #Change these blocks, fully connected then implement
        self.block1 = STGCNBlock(num_features, dim_h, 5)
        self.block2 = STGCNBlock(dim_h, dim_h, 5)
        self.block3 = STGCNBlock(dim_h, int(dim_h/2), 5)
        self.fc1 = Linear(int(dim_h/2) * num_nodes, 16 * num_nodes)
        self.b1 = BatchNorm1d(16 * num_nodes)
        self.fc2 = Linear(16 * num_nodes, num_classes)

    def forward(self, x, edge_index, train):
        x1 = self.block1(x, edge_index)
        x1 = F.dropout(x1, p=0.2, training=train)
        x2 = self.block2(x1, edge_index)
        x2 = F.dropout(x2, p=0.2, training=train)
        x3 = self.block2(x2, edge_index)
        x2 = x2.view(x2.size(0), -1)  # Flatten the tensor
        x = x1 + x3  # Skip connection
        x = F.relu(self.fc1(x))
        x = self.b1(x)
        x = F.dropout(x, p=0.1, training=train)
        x = self.fc2(x)
        return F.sigmoid(x)
    
class MultiInputSTGACN(torch.nn.Module):
    def __init__(self, num_nodes, dim_in, dim_h, num_classes, n_inputs, hcf = False):
        super(MultiInputSTGACN, self).__init__()
        #Change these blocks, fully connected then implement
        #self.block1 = STGCNBlock(dim_in, dim_h, 5)
        #self.block2 = STGCNBlock(dim_h, dim_h, 5)
        #self.block3 = STGCNBlock(dim_h, int(dim_h/2), 5)


        dim_half = int(dim_h/2)
        dim_4th = int(dim_half/2)
        dim_8th = int(dim_4th/2)
        self.num_inputs = n_inputs
        self.streams = []
        self.hcf = hcf

        for i in range(self.num_inputs):
            i_stream = []
            if i == len(range(self.num_inputs)) - 1 and hcf == True :
                print("Here?") # This is a normal linear net for HCF data without graphical structure
                i_stream.append(Linear(dim_in, dim_h))
                i_stream.append(BatchNorm1d(dim_h))
                i_stream.append(Linear(dim_h, dim_half))
                i_stream.append(BatchNorm1d(dim_half))
                i_stream.append(Linear(dim_half, dim_4th))
                i_stream.append(BatchNorm1d(dim_4th))
                i_stream.append(Linear(dim_4th, dim_8th))
                i_stream.append(BatchNorm1d(dim_8th))
            else:
                i_stream.append(STGCNBlock(dim_in, dim_h, 5))
                i_stream.append(STGCNBlock(dim_h, dim_h, 5))
                i_stream.append(STGCNBlock(dim_h, int(dim_h/2), 5))

            self.streams.append(i_stream)
        
        print("number of streams built: ", len(self.streams))
        #Final processing after combination
        self.fc1 = Linear(int(dim_h/2) * num_nodes, 16 * num_nodes)
        self.b1 = BatchNorm1d(16 * num_nodes)
        self.fc2 = Linear(16 * num_nodes, num_classes)

    def forward(self, data, edge_indices, batches, train):
        for i, x in enumerate(data):
            data[i] = data[i].to("cuda")
            if edge_indices[i] != None:
                edge_indices[i] = edge_indices[i].to("cuda")
            batches[i] = batches[i].to("cuda")
        #print("original : ", x.shape)
        #x = self.norm(x)
        hidden_layers = []
        stream_outputs = []
        h = x
        for stream_no, stream in enumerate(self.streams):
            h = data[stream_no]
            for i, layer in enumerate(stream):
                #This is the usual convolution block #if hcf, this is just a linear layer
                if i == len(stream) - 1 and self.hcf:
                    #print("going in here?")
                    if i % 2 == 0:
                        h = F.relu(layer(h))
                        h = stream[i + 1](h).to("cuda")
                else:
                    #print("only in here")
                    h = F.relu(layer(h, edge_indices[stream_no]))
                h = F.dropout(h, p=0.1, training=train)
                #Record each  hidden layer value
                hidden_layers.append(h)
            
            global_pooled = []
            for i in range(len(hidden_layers)):
                global_pooled.append(global_add_pool(hidden_layers[i], batches[i]))
            #After the stream is done, concatenate each streams layers
            #h1 = global_add_pool(hidden_layers[0], batches[0])
            #h2 = global_add_pool(hidden_layers[1], batches[1])
            #h3 = global_add_pool(hidden_layers[2], batches[2])
            #h4 = global_add_pool(hidden_layers[3], batches[3])

            # Concatenate graph embeddings
            #print("sizes: ", h1.shape, h2.shape, h3.shape)#, h4.shape)
            stream_outputs.append(torch.cat(([g for g in global_pooled]), dim=1))

        #Concatenate all stream outputs
        h_out = stream_outputs[0]
        for i, output in enumerate(stream_outputs):
            if i != 0:
                h_out = torch.cat((h_out, output), dim=1)

        print("shape of concatenated output: ", h_out.shape)

        # Classifier
        h = F.relu(self.fc1(h))
        h = self.b1(h)
        h = F.dropout(h, p=0.1, training=train)
        h = self.fc2(h)
  
        return F.sigmoid(h), F.log_softmax(h, dim=1)