import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, GINConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from Programs.Machine_Learning.Model_Based.GCN.Dataset_Obj import *

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