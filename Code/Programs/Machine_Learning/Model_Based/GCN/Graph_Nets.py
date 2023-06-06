import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, GINConv, GATv2Conv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, AvgPool2d, AvgPool1d
from Programs.Machine_Learning.Model_Based.GCN.Dataset_Obj import *

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

    def forward(self, x, edge_index, batch, train):
        x = x.to("cuda")
        edge_index = edge_index.to("cuda")
        batch = batch.to("cuda")
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


def train(model, loader, val_loader, test_loader):
    #print("types: ", type(model), type(loader), type(val_loader))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.001,
                                weight_decay=0.00005)
    epochs = 250
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

        # Train on batches
        #
        # print("data loader info: ", type(loader))
        for data in loader:
            optimizer.zero_grad()
            data = data.to("cuda")
            h, out = model(data.x, data.edge_index, data.batch, train=True)

            loss = criterion(out, data.y)
            total_loss += loss / len(loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
            loss.backward()
            optimizer.step()

            embeddings.append(h)
            losses.append(loss)
            accuracies.append(acc)
            outputs.append(out.argmax(dim=1))
            hs.append(h)


        # Validation
        val_loss, val_acc = test(model, val_loader, train = False)
        val_accs.append(val_acc)

        # Print metrics every 10 epochs
        if (epoch % 1 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
                f'| Train Acc: {acc * 100:>5.2f}% '
                f'| Val Loss: {val_loss:.2f} '
                f'| Val Acc: {val_acc * 100:.2f}%')

    if test_loader != None:
        test_loss, test_acc = test(model, test_loader)
        print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')
        test_accs.append(test_acc)
    #return model
    #print("returned lens: ", len(embeddings[0]), len(losses), len(accuracies), len(outputs), len(hs))
    return model, embeddings, losses, accuracies, outputs, val_accs, test_accs

@torch.no_grad()
def test(model, loader, train = False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    for data in loader:
        data = data.to("cuda")
        _, out = model(data.x, data.edge_index, data.batch, train)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

    return loss, acc
