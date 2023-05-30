import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

from Programs.Machine_Learning.Model_Based.GCN.Dataset_Obj import get_COO_matrix

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
    G = process_data_to_graph(data, get_COO_matrix())
    print("nodes: ", G.nodes(G))
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True, node_size=800)
    plt.show()

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
    for j, point in enumerate(train_loader):
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
        
        break

    ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2],
           s=200, c=cols, cmap="hsv", vmin=-2, vmax=3)
    plt.title(f'Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%',
              fontsize=18, pad=40)

def run_3d_animation(fig, fargs):
        plt.axis('off')
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        anim = animation.FuncAnimation(fig, animate_alt, \
                                    np.arange(0, 200, 10), interval=800, repeat=True, fargs=fargs)
        html = HTML(anim.to_html5_video())

        plt.show()
        display(html)
            
def process_data_to_graph(row, coo_matrix):
    G = nx.Graph()

    #Add nodes
    for i, x in enumerate(row.x.numpy()):
        G.add_node(int(i), pos=(-x[1], x[0]))
        #Break to avoid reading edge indices
        #break
    
    #Add edges
    for connection in joint_connections:
        G.add_edge(connection[0], connection[1])

    return G

