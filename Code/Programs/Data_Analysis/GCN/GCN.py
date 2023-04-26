import torch
import matplotlib.pyplot as plt
import torch_geometric


from Dataset_Obj import *
from Graph_Nets import GCN, GIN, GAT, train, accuracy
from Render import *
from Ground_Truths import *

def run_GCN_training():
    print(f"Torch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Torch geometric version: {torch_geometric.__version__}")

   
    #Create dataset
    dataset = JointDataset('./', 'MPI_pixels_omit_relative.csv').shuffle()
    assess_data(dataset)

    train_loader, val_loader, test_loader = create_dataloaders(dataset)

    #Define model
    print("Creating model: ")
    gin_model = GIN(dim_h=16, dataset=dataset)
    gcn_model = GIN(dim_h=16, dataset=dataset)
    gat_model = GIN(dim_h=16, dataset=dataset)

    #Train model
    #embeddings, losses, accuracies, outputs, hs = model.train(model, criterion, optimizer, data)
    print("GCN MODEL")
    model, embeddings, losses, accuracies, outputs, hs = train(gcn_model, train_loader, val_loader, test_loader)
    print("GAT MODEL") 
    model, embeddings, losses, accuracies, outputs, hs = train(gat_model, train_loader, val_loader, test_loader)
    print("GIN MODEL")
    model, embeddings, losses, accuracies, outputs, hs = train(gin_model, train_loader, val_loader, test_loader)

    # Train TSNE
    '''
    tsne = TSNE(n_components=2, learning_rate='auto',
            init='pca').fit_transform(embeddings[7].detach())

    # Plot TSNE
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=dataset[7].y)
    plt.show()
    '''

    #Animate results
    print("training complete, animating")
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    run_3d_animation(fig, (embeddings, dataset, losses, accuracies, ax, train_loader))

if __name__ == "__main__":
    #run_ground_truths()

    run_GCN_training()
