import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, GINConv, GATv2Conv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
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
    def __init__(self, dim_in, dim_h, dim_out, heads=[8,1,1,1], n_layers = 4):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads[0])
        self.gat2 = GATv2Conv(dim_h*heads[0], dim_h, heads=heads[1])
        self.gat3 = GATv2Conv(dim_h*heads[1], dim_h, heads=heads[2])
        self.gat4 = GATv2Conv(dim_h*heads[2], dim_h, heads=heads[3])
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.005,
                                          weight_decay=5e-4)
        

        out = ((n_layers -1) * dim_h) + (dim_h * heads[0])


        #for layer in range(n_layers-1)


        self.lin1 = Linear(out, out)
        self.lin2 = Linear(out, dim_out)

    def forward(self, x, edge_index, batch):
        x = x.to("cuda")
        edge_index = edge_index.to("cuda")
        batch = batch.to("cuda")
        #print("original : ", x.shape)
        h1 = F.dropout(x, p=0.6, training=self.training)
        #print("pos a : ", h1.shape, type(self.gat1))
        h1 = self.gat1(h1, edge_index)
        h1 = F.elu(h1)
        #print("pos b : ", h1.shape)

        h2 = F.dropout(h1, p=0.6, training=self.training)
        h2 = self.gat2(h2, edge_index)
        #print("pos c: ", h2.shape)

        h3 = F.dropout(h2, p=0.6, training=self.training)
        h3 = self.gat3(h3, edge_index)
        #print("pos d: ", h3.shape)

        h4 = F.dropout(h3, p=0.6, training=self.training)
        h4 = self.gat3(h4, edge_index)
        #print("pos e: ", h4.shape)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        h4 = global_add_pool(h4, batch)

        # Concatenate graph embeddings
        #print("sizes: ", h1.shape, h2.shape, h3.shape, h4.shape)
        h = torch.cat((h1, h2, h3, h4), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h, F.log_softmax(h, dim=1)
    
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
        h1 = self.gcn1(h1, edge_index)
        print("h1 shape here, ", h1.shape)
        h2 = F.dropout(h1, p=0.6, training=self.training)
        h2 = self.gcn2(h2, edge_index)

        print("h1 and 2: ", h1.shape, h2.shape)
        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        # Concatenate graph embeddings
        h = torch.cat((h1, h2), dim=1)

        # Classifier
        print("H dimensions: ", h.shape)
        tits = 5/0
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h, F.log_softmax(h, dim=1)
    
# Calculate accuracy
def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)


def train(model, loader, val_loader, test_loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.01,
                                weight_decay=0.01)
    epochs = 50
    model.train()

    # Data for animations
    embeddings = []
    losses = []
    accuracies = []
    outputs = []
    hs = []
    
    for epoch in range(epochs + 1):
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0

        # Train on batches
        print("data loader info: ")
        for data in loader:
            optimizer.zero_grad()
            data = data.to("cuda")
            h, out = model(data.x, data.edge_index, data.batch)

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
            val_loss, val_acc = test(model, val_loader)

        # Print metrics every 10 epochs
        if (epoch % 1 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
                f'| Train Acc: {acc * 100:>5.2f}% '
                f'| Val Loss: {val_loss:.2f} '
                f'| Val Acc: {val_acc * 100:.2f}%')

    test_loss, test_acc = test(model, test_loader)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')

    #return model
    print("returned lens: ", len(embeddings[0]), len(losses), len(accuracies), len(outputs), len(hs))
    return model, embeddings, losses, accuracies, outputs, hs

@torch.no_grad()
def test(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    for data in loader:
        data = data.to("cuda")
        _, out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

    return loss, acc
