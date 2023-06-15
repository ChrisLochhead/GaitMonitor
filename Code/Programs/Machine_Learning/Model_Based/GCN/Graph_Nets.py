import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, GINConv, GATv2Conv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, AvgPool2d, AvgPool1d
from Programs.Machine_Learning.Model_Based.GCN.Dataset_Obj import *
from torch_geometric.nn import ChebConv

joint_connections = [[15, 13], [13, 11], # left foot to hip 
                     [16, 14], [14, 12], # right foot to hip
                     [11, 0], [12, 0], # hips to origin
                     [9, 7], [7, 5], # left hand to shoulder
                     [10, 8], [6, 8], #right hand to shoulder
                     [5, 0], [6, 0], # Shoulders to origin
                     [1, 3], [2, 4], # ears to eyes
                     [3, 0], [4, 0],
                     [5, 6], [11, 12]]# shoulders connected, hips connected

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


class STGCNBlock(torch.nn.Module):
    def __init__(self, in_channels, dim_h, temporal_kernel_size):
        super(STGCNBlock, self).__init__()
        self.temporal_conv1 = torch.nn.Conv2d(in_channels, dim_h, kernel_size=(temporal_kernel_size, 1))
        self.spatial_conv = GATv2Conv(dim_h, dim_h, heads=1)
        #self.spatial_conv = ChebConv(dim_h, dim_h, K=3)
        self.temporal_conv2 = torch.nn.Conv2d(dim_h, in_channels, kernel_size=(temporal_kernel_size, 1))
        self.relu = ReLU()
        self.pooling = torch.nn.MaxPool2d(kernel_size=(1, 0.5))

    def forward(self, x, edge_index):
        x = self.relu(self.temporal_conv1(x))
        x = self.relu(self.spatial_conv(x, edge_index))
        x = self.relu(self.temporal_conv2(x))
        return x

class STGCN(torch.nn.Module):
    def __init__(self, num_nodes, num_features, num_classes):
        super(STGCN, self).__init__()
        #Change these blocks, fully connected then implement
        self.block1 = STGCNBlock(num_features, 64, 9)
        self.block2 = STGCNBlock(64, 64, 9)
        self.fc = Linear(64 * num_nodes, num_classes)

    def forward(self, x, edge_index):
        x1 = self.block1(x, edge_index)
        x2 = self.block2(x1, edge_index)
        x2 = x2.view(x2.size(0), -1)  # Flatten the tensor
        x = x1 + x2  # Skip connection
        x = self.fc(x)
        return x
    
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
                i_stream.append(GATv2Conv(dim_in, dim_h, heads=heads[0]))
                i_stream.append(BatchNorm1d(dim_h))
                i_stream.append(GATv2Conv(dim_h*heads[1], dim_half, heads=heads[2]))
                i_stream.append(BatchNorm1d(dim_half))
                i_stream.append(GATv2Conv(dim_half*heads[2], dim_4th, heads=heads[3]))
                i_stream.append(BatchNorm1d(dim_4th))
                i_stream.append(GATv2Conv(dim_4th*heads[3], dim_8th, heads=heads[3]))
                i_stream.append(BatchNorm1d(dim_8th))
            self.streams.append(i_stream)
        
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
            data[i] = data.to("cuda")
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

                #Only stop at each GATConv Layer
                if i / 2 == 0:
                    #This is the usual convolution block #if hcf, this is just a linear layer
                    if i == len(stream) - 1 and self.hcf:
                        print("going in here?")
                        h = F.relu(layer(h))
                    else:
                        print("only in here")
                        h = F.relu(layer(h, edge_indices[stream_no]))
                    #Batch norm always the next one
                    h = stream[i + 1](h)
                    h = F.dropout(h, p=0.1, training=train)
                    #Record each  hidden layer value
                    hidden_layers.append(h)
            
            #After the stream is done, concatenate each streams layers
            h1 = global_add_pool(hidden_layers[0], batches[0])
            h2 = global_add_pool(hidden_layers[1], batches[1])
            h3 = global_add_pool(hidden_layers[2], batches[2])
            h4 = global_add_pool(hidden_layers[3], batches[3])

            # Concatenate graph embeddings
            #print("sizes: ", h1.shape, h2.shape, h3.shape)#, h4.shape)
            stream_outputs.append(torch.cat((h1, h2, h3, h4), dim=1))

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
    
class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, dim_h, dataset):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(dataset.num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h * 3, dim_h * 3)
        self.lin2 = Linear(dim_h * 3, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h, F.log_softmax(h, dim=1)


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
        print("X SHAPE; ", x.shape)
        h1 = F.dropout(x, p=0.6, training=self.training)
        h1 = F.relu(self.gcn1(h1, edge_index))
        print("h1 shape here, ", h1.shape)
        h2 = F.dropout(h1, p=0.6, training=self.training)
        h2 = F.relu(self.gcn2(h2, edge_index))

        print("h1 and 2: ", h1.shape, h2.shape)
        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        # Concatenate graph embeddings
        h = torch.cat((h1, h2), dim=1)

        # Classifier
        print("H dimensions: ", h.shape)
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h, F.log_softmax(h, dim=1)
    
# Calculate accuracy
def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)


def train(model, loader, val_loader, test_loader, generator):
    init = generator.get_state()
    #print("types: ", type(model), type(loader), type(val_loader))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.0001,
                                weight_decay=0.00005)
    epochs = 150
    model.train()

    # Data for animations
    embeddings = []
    losses = []
    accuracies = []
    val_accs = []
    outputs = []
    hs = []
    test_accs = []
    
    for epoch in range(epochs + 1):
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0

        #First pass, append all the data together into arrays
        xs_batch = [[] for l in range(len(loader))]
        indice_batch = [[] for l in range(len(loader))]
        batch_batch = [[] for l in range(len(loader))]
        #print("lens: ", len(xs_batch))

        for i, load in enumerate(loader): 
            generator.set_state(init)
            for j, data in enumerate(load):
                data = data.to("cuda")
                #print("size here", data.x.size())
                xs_batch[i].append(data.x)
                indice_batch[i].append(data.edge_index)
                batch_batch[i].append(data.batch)

        #Second pass: process the data 
        generator.set_state(init)
        for index, data in enumerate(loader[0]):

            optimizer.zero_grad()
            data = data.to("cuda")

            #h, out = model([data.x], [data.edge_index], [data.batch], train=True)

            data_x = [xs_batch[i][index] for i in range(len(loader))]
            data_i = [indice_batch[i][index] for i in range(len(loader))]
            data_b = [batch_batch[i][index] for i in range(len(loader))]
            #print("len: ", len(data_x))
            h, out = model(data_x, data_i, data_b, train=True)
            #h, out = model([xs_batch[0][index]], [indice_batch[0][index]], [batch_batch[0][index]], train=True)

            #First data batch with Y has to have the right outputs
            loss = criterion(out, data.y)
            total_loss += loss / len(loader[0])
            #print("train outputs: ", data.y)
            acc += accuracy(out.argmax(dim=1), data.y) / len(loader[0])
            loss.backward()
            optimizer.step()

            embeddings.append(h)
            losses.append(loss)
            accuracies.append(acc)
            outputs.append(out.argmax(dim=1))
            hs.append(h)



        # Validation
        val_loss, val_acc = test(model, val_loader, generator, train = False)
        val_accs.append(val_acc)

        # Print metrics every 10 epochs
        if (epoch % 5 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
                f'| Train Acc: {acc * 100:>5.2f}% '
                f'| Val Loss: {val_loss:.2f} '
                f'| Val Acc: {val_acc * 100:.2f}%')

    if test_loader != None:
        test_loss, test_acc = test(model, test_loader, generator)
        print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')
        test_accs.append(test_acc)
    #return model
    #print("returned lens: ", len(embeddings[0]), len(losses), len(accuracies), len(outputs), len(hs))
    return model, embeddings, losses, accuracies, outputs, val_accs, test_accs

@torch.no_grad()
def test(model, loaders, generator, train = False):
    init = generator.get_state()
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    #First pass, append all the data together into arrays
    xs_batch = [[] for l in range(len(loaders))]
    indice_batch = [[] for l in range(len(loaders))]
    batch_batch = [[] for l in range(len(loaders))]
    #print("lens: ", len(xs_batch))

    for i, load in enumerate(loaders): 
        generator.set_state(init)
        for j, data in enumerate(load):
            data = data.to("cuda")
            #print("size here", data.x.size())
            xs_batch[i].append(data.x)
            indice_batch[i].append(data.edge_index)
            batch_batch[i].append(data.batch)

    #Second pass: process the data 
    generator.set_state(init)

    for index, data in enumerate(loaders[0]):
        data = data.to("cuda")

        data_x = [xs_batch[i][index] for i in range(len(loaders))]
        data_i = [indice_batch[i][index] for i in range(len(loaders))]
        data_b = [batch_batch[i][index] for i in range(len(loaders))]

        #_, out = model(data.x, data.edge_index, data.batch, train)
        _, out = model(data_x, data_i, data_b, train)
        loss += criterion(out, data.y) / len(loaders[0])
        #print("acc: ", acc, data.y)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loaders[0])

    return loss, acc
