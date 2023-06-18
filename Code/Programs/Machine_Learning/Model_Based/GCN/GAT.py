import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, GATv2Conv
from torch.nn import Linear, BatchNorm1d, AvgPool1d
from Programs.Machine_Learning.Model_Based.GCN.Dataset_Obj import *

class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, dim_in, dim_h, dim_out, heads=[1,1,1,1], n_layers = 4):
        super().__init__()
        dim_half = int(dim_h/2)
        dim_4th = int(dim_half/2)
        dim_8th = int(dim_4th/2)

        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads[0])
        self.m1 = BatchNorm1d(dim_h)
        #self.gat2 = GATv2Conv(dim_h*heads[0], dim_h, heads=heads[1])
        self.gat2 = GATv2Conv(dim_h*heads[1], dim_half, heads=heads[2])
        self.m2 = BatchNorm1d(dim_half)
        self.gat3 = GATv2Conv(dim_half*heads[2], dim_4th, heads=heads[3])
        self.m3 = BatchNorm1d(dim_4th)
        self.gat4 = GATv2Conv(dim_4th*heads[3], dim_8th, heads=heads[3])
        self.m4 = BatchNorm1d(dim_8th)
        self.m = AvgPool1d(3, stride=3, padding=1)
        self.norm = BatchNorm1d(3)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.005,
                                          weight_decay=5e-4)

        self.lin1 = Linear(256, 128)
        self.m5 = BatchNorm1d(128)
        self.lin2 = Linear(128, dim_out)

    def forward(self, xs, edge_indices, batches, train):
        #This will be passed as a 1D array if a normal GAT with only a single input stream
        x = xs[0].to("cuda")
        print("type: ", xs[0], type(xs[0]))
        stop = 5/0
        edge_index = edge_indices[0].to("cuda")
        batch = batches[0].to("cuda")
        #print("original : ", x.shape)
        #x = self.norm(x)

        h1 = F.relu(self.gat1(x, edge_index))
        h1 = self.m1(h1)
        h1 = F.dropout(h1, p=0.1, training=train)
        #print("after gat1 : ", h1.shape)

        h2 = F.relu(self.gat2(h1, edge_index))
        h2 = self.m2(h2)
        h2 = F.dropout(h2, p=0.1, training=train)
        #print("after gat2 : ", h2.shape)


        h3 = F.relu(self.gat3(h2, edge_index))
        h3 = self.m3(h3)
        h3 = F.dropout(h3, p=0.1, training=train)
        #print("after gat3: ", h3.shape)

        h4 = F.relu(self.gat4(h3, edge_index))
        h4 = self.m4(h4)
        h4 = F.dropout(h3, p=0.1, training=train)
        #print("after gat4: ", h4.shape)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        h4 = global_add_pool(h4, batch)

        # Concatenate graph embeddings
        #print("sizes: ", h1.shape, h2.shape, h3.shape)#, h4.shape)
        h = torch.cat((h1, h2, h3, h4), dim=1)

        # Classifier
        #h = self.m(h)
        h = F.relu(self.lin1(h))
        h = self.m5(h)
        h = F.dropout(h, p=0.1, training=train)
        h = self.lin2(h)

        return F.sigmoid(h), F.log_softmax(h, dim=1)


class MultiInputGAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, dim_in, dim_h, dim_out, heads=[1,1,1,1], n_inputs = 2, hcf = False):
        super().__init__()
        dim_half = int(dim_h/2)
        dim_4th = int(dim_half/2)
        dim_8th = int(dim_4th/2)
        self.num_inputs = n_inputs
        self.streams = []
        self.hcf = hcf

        for i in range(self.num_inputs):
            i_stream = []
            if i == len(range(self.num_inputs)) - 1 and hcf == True:
                print("Building HCF module: ,", i) # This is a normal linear net for HCF data without graphical structure
                i_stream.append(Linear(dim_in, dim_h))
                i_stream.append(BatchNorm1d(dim_h))
                i_stream.append(Linear(dim_h, dim_half))
                i_stream.append(BatchNorm1d(dim_half))
                i_stream.append(Linear(dim_half, dim_4th))
                i_stream.append(BatchNorm1d(dim_4th))
                i_stream.append(Linear(dim_4th, dim_8th))
                i_stream.append(BatchNorm1d(dim_8th))
            else:
                print("Building GAT module: ,", i)
                i_stream.append(GATv2Conv(dim_in, dim_h, heads=heads[0]))
                i_stream.append(BatchNorm1d(dim_h))
                i_stream.append(GATv2Conv(dim_h*heads[1], dim_half, heads=heads[2]))
                i_stream.append(BatchNorm1d(dim_half))
                i_stream.append(GATv2Conv(dim_half*heads[2], dim_4th, heads=heads[3]))
                i_stream.append(BatchNorm1d(dim_4th))
                i_stream.append(GATv2Conv(dim_4th*heads[3], dim_8th, heads=heads[3]))
                i_stream.append(BatchNorm1d(dim_8th))
            self.streams.append(i_stream)
        
        #Send to the GPU
        for i, layer_stream in enumerate(self.streams):
            for j, layer in enumerate(layer_stream):
                self.streams[i][j] = self.streams[i][j].to("cuda")

                
        
        print("number of streams built: ", len(self.streams))

        self.m = AvgPool1d(3, stride=3, padding=1)
        self.norm = BatchNorm1d(3)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.005,
                                          weight_decay=5e-4)

        #Extra linear layer to compensate for more data
        self.lin1 = Linear(256 * n_inputs, 128 * n_inputs)
        self.m1 = BatchNorm1d(128* n_inputs)
        self.lin2 = Linear(128* n_inputs, 64 * n_inputs)
        self.m2 = BatchNorm1d(64* n_inputs)
        self.lin3 = Linear(64* n_inputs, dim_out)

    def forward(self, data, edge_indices, batches, train):
        for i, x in enumerate(data):
            data[i] = data[i].to("cuda")
            #HCF info won't have edges, only joint info
            if edge_indices[i] is not None:
                edge_indices[i] = edge_indices[i].to("cuda")
            batches[i] = batches[i].to("cuda")
        #print("original : ", x.shape)

        hidden_layers = []
        stream_outputs = []
        h = data[i]
        for stream_no, stream in enumerate(self.streams):
            print("all streams: ", self.streams)
            h = data[stream_no]
            for i, layer in enumerate(stream):
                #Only stop at each GATConv Layer
                if i / 2 == 0:
                    #This is the usual convolution block #if hcf, this is just a linear layer
                    print("layer: ", i, layer, stream_no)
                    if stream_no == len(self.streams) - 1 and self.hcf:
                        h = F.relu(layer(h))
                    else:
                        print("layer: ", layer, i, self.hcf)
                        h = F.relu(layer(h, edge_indices[stream_no]))
                    #Batch norm always the next one
                    h = stream[i + 1](h)
                    h = F.dropout(h, p=0.1, training=train)
                    #Record each  hidden layer value
                    print("length of stream", i, len(stream))
                    hidden_layers.append(h)
            
            #After the stream is done, concatenate each streams layers
            h_layers = []
            for i, hidden in enumerate(hidden_layers):
                h_layers.append(global_add_pool(hidden_layers[i], batches[i]))

            #h1 = global_add_pool(hidden_layers[0], batches[0])
            #h2 = global_add_pool(hidden_layers[1], batches[1])
            #h3 = global_add_pool(hidden_layers[2], batches[2])
            #h4 = global_add_pool(hidden_layers[3], batches[3])

            # Concatenate graph embeddings
            #print("sizes: ", h1.shape, h2.shape, h3.shape)#, h4.shape)
            stream_outputs.append(torch.cat(([l for l in h_layers]), dim=1))

        #Concatenate all stream outputs
        h_out = stream_outputs[0]
        for i, output in enumerate(stream_outputs):
            if i != 0:
                h_out = torch.cat((h_out, output), dim=1)

        print("shape of concatenated output: ", h_out.shape)

        # Classifier
        h = F.relu(self.lin1(h))
        h = self.m1(h)
        h = F.dropout(h, p=0.1, training=train)
        h = F.relu(self.lin2(h))
        h = self.m2(h)
        h = F.dropout(h, p=0.1, training=train)
        h = self.lin3(h)

        return F.sigmoid(h), F.log_softmax(h, dim=1)