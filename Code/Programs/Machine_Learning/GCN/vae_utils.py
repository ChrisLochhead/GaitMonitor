'''
This file contains all the utility methods related to GCN and machine learning functions
'''
#imports
import copy
import time
import torch
torch.manual_seed(42)
import numpy as np
from torch_geometric.loader import DataLoader as GeoLoader
from torch.utils.data import RandomSampler
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
#dependencies
import Programs.Data_Processing.Utilities as Utilities
import Programs.Data_Processing.Dataset_Creator as Creator

#utility function to get accuracy from array of predictions
def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)
    

def cross_valid(m_model, test_dataset, datasets=None,k_fold=3, batch = 16,
                 epochs = 100, make_loaders = False, device = 'cuda'):
    '''
    creates a series of datasets from individual person datasets

    Arguments
    ---------
    m_model: Torch.Model()
        Machine learning model to be copied at every fold 
    test_dataset: Torch.Dataset()
        test set
    datasets: List(Torch.Model())
        train and validation datasets
    k_fold: int (optional, default = 3)
        number of folds for validation
    batch: int (optional, default = 3)
        batch size
    epochs: int (optional, default = 100)
        number of epochs per fold
    make_loaders: bool (optional, default = False)
        indicates whether to save the dataloaders
    device: str (optional, default = 'cuda')
        indicates device, either 'cpu' or 'cuda'
    
    Returns
    -------
    Torch.Model, List(), List(), List()
        returns the model and the various scoring metrics
    '''
    train_score = []
    val_score = []
    test_score = []

    # Extract the class labels
    class_labels = [data_point.y.item() for data_point in datasets[0]]
    # Initialize the StratifiedKFold object
    skf = StratifiedKFold(n_splits=k_fold)
    folded_train = [[] for l in range(len(datasets))]
    folded_val = [[] for l in range(len(datasets))]

    for train_indices, val_indices in skf.split(datasets[0], class_labels):
        for d_ind, dataset in enumerate(datasets):
            train_data = [dataset[i] for i in train_indices]
            val_data = [dataset[i] for i in val_indices]

            folded_train[d_ind].append(copy.deepcopy(train_data))
            folded_val[d_ind].append(copy.deepcopy(val_data))

    total_preds = []
    total_ys = []
    for fold in range(k_fold):
        print(f"Fold: {fold} of {k_fold}")
        start = time.time()
        train_loaders = []
        val_loaders = []
        test_loaders = []

        for i, dataset in enumerate(datasets):
            G = torch.Generator()
            train_sample = RandomSampler(folded_train[i][fold], generator=G,)
            #Restrict validation and testing batch sizes to be one batch
            val_sample = RandomSampler(folded_val[i][fold], generator=G)
            test_sample = RandomSampler(test_dataset[i], generator=G)

            test_set = test_dataset[i]
            train_set = folded_train[i][fold]
            val_set = folded_val[i][fold]

            train_loaders.append(GeoLoader(train_set, batch_size=batch, sampler=train_sample, drop_last = True))
            #Restrict val set to only being 1 batch, so the same batch and hence the same data is always picked for testing
            val_loaders.append(GeoLoader(val_set, batch_size=batch, sampler = val_sample, drop_last = True))
            test_loaders.append(GeoLoader(test_set, batch_size=batch, sampler = test_sample, drop_last = True))

            if make_loaders:
                return train_loaders, val_loaders, test_loaders
        #reset the model, train and save the results to the result lists
        model = copy.deepcopy(m_model)
        model = model.to(device)
        model, accuracies, vals, tests, all_y, all_pred = train(model, train_loaders, val_loaders, test_loaders, G, epochs, device)
        total_ys += all_y
        total_preds += all_pred
        train_score.append(accuracies[-1])
        val_score.append(vals[-1])
        test_score.append(tests[-1])
        end = time.time()
        print("time elapsed: ",fold,  end - start)

    print("final confusion: ")
    print(confusion_matrix(total_ys, total_preds))
    f1 = f1_score(total_ys, total_preds, average='weighted')
    print("f1 score: ", f1)
    return model, train_score, val_score, test_score


def vae_loss(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(model, loader, val_loader, test_loader, generator, epochs, device):
    '''
    creates a series of datasets from individual person datasets

    Arguments
    ---------
    model: Torch.Model()
        Machine learning model
    loader: Torch.Dataloader()
        train loader object
    val_loader: Torch.Dataloader()
        val loader object
    test_loader: Torch.Dataloader()
        test loader object
    generator: Torch.Generator
        generator object for consistent seed generation across datasets
    epochs: int (optional, default = 100)
        number of epochs per fold
    device: str (optional, default = 'cuda')
        indicates device, either 'cpu' or 'cuda'
    
    Returns
    -------
    Torch.Model, List(), List(), List(), List(), List()
        returns the model and the various scoring metrics
    '''
    init = generator.get_state()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.1,
                                weight_decay=0.001)
    model.train()

    #First pass, append all the data together into arrays
    xs_batch = [[] for l in range(len(loader))]
    indice_batch = [[] for l in range(len(loader))]
    batch_batch = [[] for l in range(len(loader))]
    ys_batch = [[] for l in range(len(loader))]

    for ind, load in enumerate(loader): 
        generator.set_state(init)
        for j, data in enumerate(load):
            data = data.to(device)
            xs_batch[ind].append(data.x)
            indice_batch[ind].append(data.edge_index)
            batch_batch[ind].append(data.batch)
            ys_batch[ind].append(data.y)  

    for epoch in range(epochs + 1):
        #Reduce by 0.1 times at 10th and 60th epoch
        if epoch == 20:
            #print("reducing learing rate")
            optimizer.param_groups[0]['lr'] = 0.01
        elif epoch == 60:
            #print("reducing learning rate again")
            optimizer.param_groups[0]['lr'] = 0.001

        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0
        #Second pass: process the data 
        generator.set_state(init)
        for index, data in enumerate(loader[0]):
            optimizer.zero_grad()
            data_x = [xs_batch[i][index] for i in range(len(loader))]
            data_i = [indice_batch[i][index] for i in range(len(loader))]
            data_b = [batch_batch[i][index] for i in range(len(loader))]

            recon_batch, mu, log_var = model(data_x, data_i, data_b, train)
            loss = vae_loss(recon_batch, data_x, mu, log_var)

            total_loss = total_loss + loss
            loss.backward()
            optimizer.step()

        # Validation
        generator.set_state(init)
        val_loss = test(model, val_loader, generator, validation=True, device = device)

        # Print metrics every 10 epochs
        if (epoch % 5 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
                f'| Train Acc: {acc * 100:>5.2f}% '
                f'| Val Loss: {val_loss:.2f} '
                f'| Val Acc: {val_acc * 100:.2f}%')

            if test_loader != None:
                generator.set_state(init)
                test_loss = test(model, test_loader, generator, validation=False, device = device)
                print(f'Test Loss: {test_loss:.2f}')

    if test_loader != None:
        generator.set_state(init)
        test_loss = test(model, test_loader, generator, validation=False, device = device)
        print(f'Test Loss: {test_loss:.2f}')

    #return model
    return model

    
def test(model, loaders, generator, validation, train = False, device = 'cuda'):
    '''
    creates a series of datasets from individual person datasets

    Arguments
    ---------
    model: Torch.Model()
        Machine learning model
    loaders: Torch.Dataloader()
        test loader object
    generator: Torch.Generator
        generator object for consistent seed generation across datasets
    validation: bool
        indicates whether this is validation or test
    train: bool (optional, default = false)
        indicates whether to adjust gradients or not in model
    device: str (optional, default = 'cuda')
        indicates device, either 'cpu' or 'cuda'
    
    Returns
    -------
    List(), List(), List(), List()
        returns the model and the various scoring metrics
    '''
    init = generator.get_state()
    model.eval()
    loss = 0
    with torch.no_grad():
        #First pass, append all the data together into arrays
        xs_batch = [[] for l in range(len(loaders))]
        indice_batch = [[] for l in range(len(loaders))]
        batch_batch = [[] for l in range(len(loaders))]
        ys_batch = [[] for l in range(len(loaders))]
        for i, load in enumerate(loaders): 
            generator.set_state(init)
            for j, data in enumerate(load):
                data = data.to(device)
                xs_batch[i].append(data.x)
                indice_batch[i].append(data.edge_index)
                batch_batch[i].append(data.batch)
                ys_batch[i].append(data.y)

        #Second pass: process the data 
        generator.set_state(init)
        total_loss = 0
        for index, data in enumerate(loaders[0]):
            data_x = [xs_batch[i][index] for i in range(len(loaders))]
            data_i = [indice_batch[i][index] for i in range(len(loaders))]
            data_b = [batch_batch[i][index] for i in range(len(loaders))]

            recon_batch, mu, log_var = model(data_x, data_i, data_b, train)
            loss = vae_loss(recon_batch, data_x, mu, log_var)
            total_loss = total_loss + loss

    return total_loss