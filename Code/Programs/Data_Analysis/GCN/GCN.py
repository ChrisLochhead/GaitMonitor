# Numpy for matrices
import numpy as np
import torch

# Visualization libraries
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import KarateClub, PascalVOCKeypoints
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_networkx
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, GINConv, GATv2Conv
import pandas as pd
from torch_geometric.data import Data
import torch_geometric
from torch_geometric.data import Dataset
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout

joint_connections = [[15, 13], [13, 11], # left foot to hip 
                     [16, 14], [14, 12], # right foot to hip
                     [11, 0], [12, 0], # hips to origin
                     [9, 7], [7, 5], # left hand to shoulder
                     [10, 8], [6, 8], #right hand to shoulder
                     [5, 0], [6, 0], # Shoulders to origin
                     [1, 3], [2, 4], # ears to eyes
                     [3, 0], [4, 0],
                     [5, 6], [11, 12]]# shoulders connected, hips connected

                    

#%%capture
from IPython.display import display, HTML
from matplotlib import animation
plt.rcParams["animation.bitrate"] = 3000
plt.rcParams['animation.ffmpeg_path'] = "C:/Users/Chris/Desktop/ffmpeg-5.1.2-full_build/bin/ffmpeg.exe"

def plot_graph(data):
    #G = to_networkx(data, to_undirected=True)
    G = process_data_to_graph(data, get_COO_matrix())
    print("nodes: ", G.nodes(G))
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True, node_size=800)
   # nx.draw_networkx(G,
   #                  pos=nx.spring_layout(G, seed=0),
   #                  with_labels=True,
   #                  node_size=800,
   #                  #node_color=data.y,
   #                  cmap="hsv",
   #                  vmin=-2,
   #                  vmax=3,
   #                  width=0.8,
   #                  edge_color="grey",
   #                  font_size=14
   #                 )
    plt.show()

class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.005,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
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
      self.gcn2 = GCNConv(dim_h, dim_out)
      self.optimizer = torch.optim.Adam(self.parameters(),
                                        lr=0.01,
                                        weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.5, training=self.training)
        h = self.gcn1(h, edge_index).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn2(h, edge_index)
        return h, F.log_softmax(h, dim=1)
    
# Calculate accuracy
def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)


def train(model, loader, val_loader, test_loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.01,
                                weight_decay=0.01)
    epochs = 1000

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
        for data in loader:
            optimizer.zero_grad()
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
        if (epoch % 10 == 0):
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
        _, out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

    return loss, acc

def animate(i, *fargs):
    data = fargs[0]
    outputs = fargs[1]
    losses = fargs[2]
    accuracies = fargs[3] 

    G = to_networkx(data, to_undirected=True)
    nx.draw_networkx(G,
                    pos=nx.spring_layout(G, seed=0),
                    with_labels=True,
                    node_size=800,
                    node_color="blue",#outputs[i],
                    cmap="hsv",
                    vmin=-2,
                    vmax=3,
                    width=0.8,
                    edge_color="grey",
                    font_size=14
                    )
    plt.title(f'Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%',
              fontsize=18, pad=20)


def animate_alt(i, *fargs):
    embeddings = fargs[0]
    data = fargs[1]
    losses = fargs[2]
    accuracies = fargs[3]
    ax = fargs[4]
    train_loader = fargs[5]

    embed = embeddings[i].detach().cpu().numpy()
    ax.clear()

    cols = []
    ite = 0
    for j, point in enumerate(train_loader):
        if i != j:
            continue
    
        for k, em in enumerate(point):
            if k == 2: 
                class_vals = em[1].numpy()
                for val in class_vals:
                    col = "blue"
                    if val == 1:
                        col = "red"
                    elif val == 2:
                        col = "green"
                    cols.append(col)

    ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2],
           s=200, c=cols, cmap="hsv", vmin=-2, vmax=3)
    plt.title(f'Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%',
              fontsize=18, pad=40)

def main():
    #Create dataset

    test = KarateClub()

    print(test.num_features)
    for i in test[0].x.numpy():
        print("i : ", i)
    print("len: ", len(test[0].x))
    dataset = JointDataset('./', 'pixel_data_absolute.csv').shuffle()
    print("Dataset type: ", type(dataset), type(dataset[0]))

    print('------------')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of nodes: {dataset[7].num_nodes}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Graph: {dataset[0]}')
    print(f'Graph2: {dataset[1]}')
    #Print individual node information
    data = dataset[7]
    print(f'x = {data.x.shape}')
    print(data.x)
    print(data.y.shape, type(data.y), data.y)


    #Not sure what these'll say
    print(f'Edges are directed: {data.is_directed()}')
    print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Graph has loops: {data.has_self_loops()}')

    #Plot graph data using networkX
    plot_graph(data)

        # Create training, validation, and test sets
    train_dataset = dataset[:int(len(dataset)*0.8)]
    val_dataset   = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
    test_dataset  = dataset[int(len(dataset)*0.9):]

    print(f'Training set   = {len(train_dataset)} graphs')
    print(f'Validation set = {len(val_dataset)} graphs')
    print(f'Test set       = {len(test_dataset)} graphs')

    # Create mini-batches
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    #Define model
    print("Creating model: ")
    
    gin_model = GIN(dim_h=16, dataset=dataset)
    gcn_model = GIN(dim_h=16, dataset=dataset)
    gat_model = GIN(dim_h=16, dataset=dataset)
    #print(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gin_model.parameters(), lr=0.02)

    #Train model
    #embeddings, losses, accuracies, outputs, hs = model.train(model, criterion, optimizer, data)
    print("GCN MODEL")
    model, embeddings, losses, accuracies, outputs, hs = train(gcn_model, train_loader, val_loader, test_loader)
    '''
    print("GAT MODEL") 
    model, embeddings, losses, accuracies, outputs, hs = train(gat_model, train_loader, val_loader, test_loader)
    print("GIN MODEL")
    model, embeddings, losses, accuracies, outputs, hs = train(gin_model, train_loader, val_loader, test_loader)
    '''

    #Animate results
    print("training complete, animating")

    
    #fig = plt.figure(figsize=(12, 12))
    #plt.axis('off')

    #print("lengths: ", len(outputs), len(losses), len(accuracies))
    #anim = animation.FuncAnimation(fig, animate, np.arange(0, 200, 10), interval=500, repeat=True, fargs=[data, outputs, losses, accuracies])
    
    #html = HTML(anim.to_html5_video())
    #plt.show()
    #display(html)
    '''
    embed = hs[0].detach().cpu().numpy()

    #Second figure for animation
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.patch.set_alpha(0)
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    
    col = (255, 0, 0)
    if data.y == 0:
        col = (0, 255, 0)
    elif data.y == 1:
        col = (0,0,255)

    ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2],
               s=200, c="blue", cmap="hsv", vmin=-2, vmax=3)

    plt.show()
    '''
    
    fig = plt.figure(figsize=(12, 12))
    plt.axis('off')
    ax = fig.add_subplot(projection='3d')
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)

    anim = animation.FuncAnimation(fig, animate_alt, \
                                   np.arange(0, 200, 10), interval=800, repeat=True, fargs=(embeddings, dataset, losses, accuracies, ax, train_loader))
    html = HTML(anim.to_html5_video())

    plt.show()
    display(html)


def shift_class_col(df):
    cols_at_end = ['Class']
    df = df[[c for c in df if c not in cols_at_end] 
            + [c for c in cols_at_end if c in df]]

    return df
def process_dataset():
    colnames=['Instance', 'No_In_Sequence', 'Class', 'Joint_1','Joint_2','Joint_3','Joint_4','Joint_5','Joint_6','Joint_7',
        'Joint_8','Joint_9','Joint_10','Joint_11','Joint_12','Joint_13','Joint_14','Joint_15','Joint_16', 'Joint_17'] #

    dataset_master = pd.read_csv("pixel_data_absolute.csv", names =colnames, header=None )
    dataset_master = shift_class_col(dataset_master)
    coo_matrix = get_COO_matrix()
    print(dataset_master.head())

    graph_dataset = []
    for index, row in dataset_master.iterrows():
        graph_dataset.append(data_to_graph(row, coo_matrix))

def get_COO_matrix():
  res = [[],[]]
  for connection in joint_connections:
    #Once for each of the 3 coords in each connection
    for i in range(0, 3):
        res[0] += [connection[0], connection[1]]
        res[1] += [connection[1], connection[0]]
  return res

import ast 
import copy 

def convert_to_literals(data):
    for i,  (index, row) in enumerate(data.iterrows()):
        for col_index, col in enumerate(row):
            if col_index >= 3:
                tmp = ast.literal_eval(row[col_index])
                data.iat[i, col_index] = copy.deepcopy(tmp)
            else:
                data.iat[i, col_index] = int(data.iat[i, col_index])

    return data

def unravel_data(row):
    processed_row = []
    for coords in row:
        for value in coords:
            processed_row.append(copy.deepcopy(value))
    
    return processed_row
            
def process_data_to_graph(row, coo_matrix):
    G = nx.Graph()

    #Add nodes
    #for i, val in enumerate(row):
    print("row", row.x)
    for i, x in enumerate(row.x.numpy()):
        #if i % 3 == 0:
        print("node", x, -x[0], x[1])
        G.add_node(int(i), pos=(-x[1], x[0]))# pos=(-row.x.numpy()[i+ 1], row.x.numpy()[i]))
        #Break to avoid reading edge indices
        #break
    
    #Add edges
    for connection in joint_connections:
        G.add_edge(connection[0], connection[1])

    return G


#Input here would be each row
def data_to_graph(row, coo_matrix):
    refined_row = row.iloc[3:]
    node_f= refined_row

    #This is standard Data that has edge shit
    row_as_array = np.array(node_f.values.tolist())

    #Turn into one-hot vector
    y = int(row.iloc[2])

    data = Data(x=torch.tensor(row_as_array, dtype=torch.float),
                y=torch.tensor([y], dtype=torch.long),
                edge_index=torch.tensor(coo_matrix, dtype=torch.long),
                #This isn't needed
                #edge_attr=torch.tensor(edge_attr,dtype=torch.float)
                )
    return data

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class JointDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        super(JointDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0], header=None)
        print("data being read from: ", self.raw_paths[0], type(self.raw_paths[0]))
        print(self.raw_paths[0])
        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
        

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0], header=None)#.reset_index()
        self.data = convert_to_literals(self.data)
        coo_matrix = get_COO_matrix()

        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            data = data_to_graph(row, coo_matrix)

            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))
            

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data
    
if __name__ == "__main__":
    main()
