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

        linear_input = dim_h + dim_half + dim_4th + dim_8th
        self.lin1 = Linear(linear_input, 64)
        self.m5 = BatchNorm1d(64)
        self.lin2 = Linear(64, dim_out)

    def forward(self, xs, edge_indices, batches, train):
        #This will be passed as a 1D array if a normal GAT with only a single input stream
        x = xs[0].to("cuda")
        edge_index = edge_indices[0].to("cuda")
        batch = batches[0].to("cuda")
        #print("original : ", x.shape)


        h1 = F.relu(self.gat1(x, edge_index))
        h1 = self.m1(h1)
        h1 = F.dropout(h1, p=0.5, training=train)
        #print("after gat1 : ", h1.shape)

        h2 = F.relu(self.gat2(h1, edge_index))
        h2 = self.m2(h2)
        h2 = F.dropout(h2, p=0.5, training=train)
        #print("after gat2 : ", h2.shape)


        h3 = F.relu(self.gat3(h2, edge_index))
        h3 = self.m3(h3)
        h3 = F.dropout(h3, p=0.5, training=train)
        #print("after gat3: ", h3.shape)

        h4 = F.relu(self.gat4(h3, edge_index))
        h4 = self.m4(h4)
        h4 = F.dropout(h4, p=0.5, training=train)
        #print("after gat4: ", h4.shape)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        h4 = global_add_pool(h4, batch)

        # Concatenate graph embeddings
        #print("sizes: ", h1.shape, h2.shape, h3.shape, h4.shape)
        h = torch.cat((h1, h2, h3, h4), dim=1)

        # Classifier
        h = F.relu(self.lin1(h))
        h = self.m5(h)
        h = F.dropout(h, p=0.1, training=train)
        h = self.lin2(h)

        return F.sigmoid(h), h

class GATResNetBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads = [1,1]):
        super(GATResNetBlock, self).__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads[0])
        self.bn1 = torch.nn.BatchNorm1d(dim_h)
        self.relu = torch.nn.ReLU(inplace=True)
        self.gat2 = GATv2Conv(dim_h, dim_out, heads=heads[0])
        self.bn2 = torch.nn.BatchNorm1d(dim_out)
        self.shortcut = GATv2Conv(dim_in, dim_out, heads=heads[0])    
        self.bn3 = torch.nn.BatchNorm1d(dim_out)

    def forward(self, x, edge_indices):
        residual = x
        residual_edge = edge_indices
        out = self.gat1(x, edge_indices)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.gat2(out, edge_indices)
        out = self.bn2(out)
        res_out = self.shortcut(residual, residual_edge)
        res_out = self.bn3(res_out)
        out += res_out  # Residual connection
        return out

class HCFNet(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=[1,1,1,1], n_inputs = 2, hcf = False):
        super().__init__()
        dim_half = int(dim_h/2)
        dim_4th = int(dim_half/2)
        dim_8th = int(dim_4th/2)

        self.stream = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_h),
            torch.nn.BatchNorm1d(dim_h),
            torch.nn.Linear(dim_h, dim_half),
            torch.nn.BatchNorm1d(dim_half),
            torch.nn.Linear(dim_half, dim_4th),
            torch.nn.BatchNorm1d(dim_4th),
            torch.nn.Linear(dim_4th, dim_8th),
            torch.nn.BatchNorm1d(dim_8th),
            torch.nn.Linear(dim_8th, dim_out)
        )

    def forward(self, data, edge_indices=None, batches=None, train=True):
        for i, x in enumerate(data):
            data[i] = data[i].to("cuda")
            h = data[i]

            for j, layer in enumerate(self.stream):
                if j != len(self.stream) - 1:
                    if j % 2 == 0:
                        h = F.relu(layer(h))
                    else:
                        h = layer(h)
                        F.dropout(h, 0.5, training=train)
                else:
                    h = layer(h)

        return torch.sigmoid(h), h

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
            if i == len(range(self.num_inputs)) - 1 and self.hcf == True:
                print("Building HCF module: ,", i, dim_in) # This is a normal linear net for HCF data without graphical structure
                
                i_stream.append(Linear(dim_in[i], dim_h))
                i_stream.append(BatchNorm1d(dim_h))
                i_stream.append(Linear(dim_h, dim_half))
                i_stream.append(BatchNorm1d(dim_half))
                i_stream.append(Linear(dim_half, dim_4th))
                i_stream.append(BatchNorm1d(dim_4th))
                i_stream.append(Linear(dim_4th, dim_8th))
                i_stream.append(BatchNorm1d(dim_8th))
            else:
                print("Building GAT module: ,", i, dim_in)
                i_stream.append(GATResNetBlock(dim_in[i], dim_h, dim_half))
                i_stream.append(GATResNetBlock(dim_half, dim_4th, dim_8th))

            self.streams.append(i_stream)
        
        #Send to the GPU
        for i, layer_stream in enumerate(self.streams):
            for j, layer in enumerate(layer_stream):
                self.streams[i][j] = self.streams[i][j].to("cuda")
        
        print("number of streams built: ", len(self.streams))

        #Extra linear layer to compensate for more data
        total_num_layers = len(self.streams)
        if self.hcf:
            linear_input = total_num_layers -1
        else:
            linear_input = total_num_layers

        linear_input = linear_input * dim_half
        #HCF only concatenates the last (or smallest) hidden layer, GAT convs take all 4 layers
        if self.hcf:
            linear_input += dim_8th

        self.lin1 = Linear(linear_input, 128)
        self.m1 = BatchNorm1d(128)
        self.lin2 = Linear(128, 64)
        self.m2 = BatchNorm1d(64)
        self.lin3 = Linear(64, dim_out)

    def forward(self, data, edge_indices, batches, train):
        for i, x in enumerate(data):
            data[i] = data[i].to("cuda")
            #HCF info won't have edges, only joint info
            if edge_indices[i] is not None:
                edge_indices[i] = edge_indices[i].to("cuda")
            batches[i] = batches[i].to("cuda")

        hidden_layers = []
        h = data[i]

        for stream_no, stream in enumerate(self.streams):
            h = data[stream_no]
            hidden_layer_stream = []
            for i, layer in enumerate(stream):
                #Only stop at each GATConv Layer
                if i % 2 == 0:
                    #This is the usual convolution block #if hcf, this is just a linear layer
                    if stream_no == len(self.streams) - 1 and self.hcf:
                        #print("hcf layer: ", layer, i, self.hcf, h.shape)
                        h = F.relu(layer(h))
                    else:
                        #print("GAT layer: ", layer, i, self.hcf, h.shape)
                        h = F.relu(layer(h, edge_indices[stream_no]))
                   
                    #Dropout always the next one
                    h = F.dropout(h, p=0.9, training=train)


                    if self.hcf and stream_no + 1 == len(self.streams):
                        if i == len(stream) - 2:
                            hidden_layer_stream.append(h)
                    else:
                        hidden_layer_stream.append(h)

            hidden_layers.append(hidden_layer_stream)
            
        #After the stream is done, concatenate each streams layers
        h_layers = []
        for i, hidden_stream in enumerate(hidden_layers):
            #print("new stream", i)
            for j, layer in enumerate(hidden_stream):
                #print("stream {} layer {}".format(i, j))
                #print("size: ", hidden_layers[i][j].shape, j)
                h_layers.append(global_add_pool(hidden_layers[i][j], batches[i]))

        # Concatenate graph embeddings
        h_out = torch.cat(([l for l in h_layers]), dim=1)

        # Classifier
        h = F.relu(self.lin1(h_out))
        h = self.m1(h)
        h = F.dropout(h, p=0.5, training=train)
        h = F.relu(self.lin2(h))
        h = self.m2(h)
        #h = F.dropout(h, p=0.1, training=train)
        h = self.lin3(h)

        return F.sigmoid(h), h