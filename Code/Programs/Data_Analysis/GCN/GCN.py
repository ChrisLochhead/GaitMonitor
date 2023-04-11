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
from torch_geometric.nn import GCNConv
import pandas as pd
from torch_geometric.data import Data
import torch_geometric
from torch_geometric.data import Dataset
import os
from tqdm import tqdm

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

class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.gcn = GCNConv(dataset.num_features, 3)
        self.out = Linear(3, dataset.num_classes)

    def forward(self, x, edge_index):
        print("forward pass: ", len(x))
        print(x)
        print("edge index: ", edge_index)
        h = self.gcn(x, edge_index).relu()
        z = self.out(h)
        return h, z

    # Calculate accuracy
    def accuracy(self, pred_y, y):
        return (pred_y == y).sum() / len(y)


    def train(self, model, criterion, optimizer, data):

        # Data for animations
        embeddings = []
        losses = []
        accuracies = []
        outputs = []
        hs = []

        # Training loop
        for epoch in range(201):
            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            h, z = model(data.x, data.edge_index)

            # Calculate loss function
            loss = criterion(z, data.y)

            # Calculate accuracy
            acc = model.accuracy(z.argmax(dim=1), data.y)

            # Compute gradients
            loss.backward()

            # Tune parameters
            optimizer.step()

            # Store data for animations
            embeddings.append(h)
            losses.append(loss)
            accuracies.append(acc)
            outputs.append(z.argmax(dim=1))
            hs.append(h)

            # Print metrics every 10 epochs
            if epoch % 10 == 0:
                print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')

        return embeddings, losses, accuracies, outputs, hs



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
                    node_color=outputs[i],
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

    embed = embeddings[i].detach().cpu().numpy()
    ax.clear()
    ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2],
           s=200, c=data.y, cmap="hsv", vmin=-2, vmax=3)
    plt.title(f'Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%',
              fontsize=18, pad=40)

def main():
    #Create dataset

    test = KarateClub()

    print(test.num_features)
    for i in test[0].x.numpy():
        print("i : ", i)
    print("len: ", len(test[0].x))
    dataset = JointDataset('./', 'pixel_data_absolute.csv')
    print("Dataset type: ", type(dataset), type(dataset[0]))

    print('------------')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
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

    #Define model
    print("Creating model: ")
    model = GCN(dataset=dataset)
    print(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    #Train model
    embeddings, losses, accuracies, outputs, hs = model.train(model, criterion, optimizer, data)

    #Animate results
    print("training complete, animating")
    fig = plt.figure(figsize=(12, 12))
    plt.axis('off')

    anim = animation.FuncAnimation(fig, animate, np.arange(0, 200, 10), interval=500, repeat=True, fargs=[data, outputs, losses, accuracies])
    
    html = HTML(anim.to_html5_video())
    plt.show()
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
    ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2],
               s=200, c=data.y, cmap="hsv", vmin=-2, vmax=3)

    plt.show()

    fig = plt.figure(figsize=(12, 12))
    plt.axis('off')
    ax = fig.add_subplot(projection='3d')
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)

    anim = animation.FuncAnimation(fig, animate_alt, \
                                   np.arange(0, 200, 10), interval=800, repeat=True, fargs=(embeddings, data, losses, accuracies, ax))
    html = HTML(anim.to_html5_video())

    plt.show()
'''


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
        if i % 3 == 0:
            G.add_node(int(i/3), pos=(-row.x.numpy()[i+ 1], row.x.numpy()[i]))
        #Break to avoid reading edge indices
     #   break
    
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
    print("row as array: ", row_as_array)
    #row_as_array = unravel_data(row_as_array)

    data = Data(x=torch.tensor([row_as_array], dtype=torch.float),
                y=torch.tensor([row.iloc[2]], dtype=torch.int),
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
