# Numpy for matrices
import numpy as np
import torch

# Visualization libraries
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_networkx
from torch.nn import Linear
from torch_geometric.nn import GCNConv
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
    dataset = KarateClub()

    # Print information for whole graph from data
    print(dataset)
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
    main()