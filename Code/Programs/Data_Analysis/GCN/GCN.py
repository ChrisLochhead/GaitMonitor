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
#%%capture
from IPython.display import display, HTML
from matplotlib import animation
plt.rcParams["animation.bitrate"] = 3000
plt.rcParams['animation.ffmpeg_path'] = "C:/Users/Chris/Desktop/ffmpeg-5.1.2-full_build/bin/ffmpeg.exe"

def plot_graph(data):
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    nx.draw_networkx(G,
                     pos=nx.spring_layout(G, seed=0),
                     with_labels=True,
                     node_size=800,
                     node_color=data.y,
                     cmap="hsv",
                     vmin=-2,
                     vmax=3,
                     width=0.8,
                     edge_color="grey",
                     font_size=14
                     )
    plt.show()

class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.gcn = GCNConv(dataset.num_features, 3)
        self.out = Linear(3, dataset.num_classes)

    def forward(self, x, edge_index):
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
    # Import dataset from PyTorch Geometric
    #dataset = KarateClub()
    dataset = PascalVOCKeypoints("./Datasets", "Person")

    # Print information for whole graph from data
    print(dataset)
    print("type: ", type(dataset), type(dataset[0]))
    print('------------')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Graph: {dataset[0]}')

    #Print individual node information
    data = dataset[0]
    print(f'x = {data.x.shape}')
    print(data.x)

    #Convert from bag of words form to dense adjacency matrix
    A = to_dense_adj(data.edge_index)[0].numpy().astype(int)
    print(f'A = {A.shape}')
    print(A)

    print(f'y = {data.y.shape}')
    print(data.y)

    print(f'train_mask = {data.train_mask.shape}')
    print(data.train_mask)

    print(f'Edges are directed: {data.is_directed()}')
    print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Graph has loops: {data.has_self_loops()}')

    #Plot graph data using networkX
    plot_graph(data)

    #Define model
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

if __name__ == "__main__":
    #Step 1: Download data:
    #colnames=['Instance', 'No_In_Sequence', 'Class', 'Joint_1','Joint_2','Joint_3','Joint_4','Joint_5','Joint_6','Joint_7',
    #      'Joint_8','Joint_9','Joint_10','Joint_11','Joint_12','Joint_13','Joint_14','Joint_15','Joint_16', 'Joint_17'] #

#    dataset_master = pd.read_csv("pixel_data_absolute.csv", names =colnames, header=None )
#    print(dataset_master.head())
    main()


#%%

import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset
import numpy as np 
import os
from tqdm import tqdm
#import deepchem as dc
#from rdkit import Chem 

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
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
        

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            #mol = Chem.MolFromSmiles(row["smiles"])
            #f = featurizer._featurize(mol)
            #data = f.to_pyg_graph()
            data = Data

            #Change data.x to one video per instance for 60 instances, each having 1 class label and 1 instance as metadata 

            data.instance = row[0]
            data.no_in_sequence = row[2]
            data.x = row.iloc[3:]
            data.y = self._get_label(row[2])


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
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data

#%%

class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt', ...]

    def download(self):
        # Download to `self.raw_dir`.
        path = download_url(url, self.raw_dir)
        ...

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = Data(...)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data