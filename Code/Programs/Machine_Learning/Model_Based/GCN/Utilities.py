import Programs.Machine_Learning.Model_Based.GCN.Render as Render
import torch
from torch_geometric.loader import DataLoader as GeoLoader
from torch.utils.data import RandomSampler
import Programs.Machine_Learning.Model_Based.GCN.GAT as gat
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import StratifiedKFold
import copy
import torch.nn.functional as F

def assess_data(dataset):
    print("Dataset type: ", type(dataset), type(dataset[0]))
    print('------------')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of nodes: {dataset[0].num_nodes}')
    print(f'Number of classes: {dataset.num_classes}')
    #Print individual node information
    data = dataset[0]
    print(f'x = {data.x.shape}')
    print(data.x)
    print(data.y.shape, type(data.y), data.y)
    print(f'Edges are directed: {data.is_directed()}')
    print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Graph has loops: {data.has_self_loops()}')
    #Plot graph data using networkX
    #Render.plot_graph(data)

    # Calculate accuracy
def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)


# define a cross validation function
def cross_valid(MY_model, test_dataset, criterion=None,optimizer=None,datasets=None,k_fold=3, batch = 16, inputs_size = 1, epochs = 100, type = "GAT"):
    
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

    for fold in range(k_fold):

        train_loaders = []
        val_loaders = []
        test_loaders = []

        #Set up so identical seed is used
        torch.manual_seed(13)
        G = torch.Generator()

        train_sample = RandomSampler(folded_train[0][fold], generator=G)

        #Restrict validation and testing batch sizes to be one batch
        if type == "ST-AGCN":
            val_sample = RandomSampler(folded_val[0][fold][0:batch], generator=G)
            test_sample = RandomSampler(test_dataset[0][0:batch], generator=G)
        else:
            val_sample = RandomSampler(folded_val[0][fold], generator=G)
            test_sample = RandomSampler(test_dataset[0], generator=G)

        init = G.get_state()
        for i, dataset in enumerate(datasets):
            test_set = test_dataset[i]
            train_set = folded_train[i][fold]
            val_set = folded_val[i][fold]

            train_loaders.append(GeoLoader(train_set, batch_size=batch, sampler = train_sample, drop_last = True))

            #ST-GCNs, these are different because ST-GCN requires padded samples of all the same size
            if type == "ST-AGCN":
                #Restrict val set to only being 1 batch, so the same batch and hence the same data is always picked for testing
                test_set = test_set[0:batch]
                val_set = val_set[0:batch]
                val_loaders.append(GeoLoader(val_set, batch_size=batch,sampler = val_sample, drop_last = True ))
                test_loaders.append(GeoLoader(test_set, batch_size=batch,sampler = test_sample, drop_last = True))
            else:
            #GATs
                val_loaders.append(GeoLoader(val_set, batch_size=len(val_set)))
                test_loaders.append(GeoLoader(test_set, batch_size=len(test_set)))

            #Reset the generator so every dataset gets the same sampling 
            G.set_state(init)

        G.set_state(init)

        model = copy.deepcopy(MY_model)
        model = model.to("cuda")
            
        model, accuracies, vals, tests = train(model, train_loaders, val_loaders, test_loaders, G, epochs)
        train_score.append(accuracies[-1])
        val_score.append(vals[-1])
        test_score.append(tests[-1])
    
    return train_score, val_score, test_score

def train(model, loader, val_loader, test_loader, generator, epochs):
    init = generator.get_state()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.01,
                                weight_decay=0.00005)
    epochs = epochs
    model.train()

    train_accs = []
    val_accs = []
    test_accs = []
    
    #First pass, append all the data together into arrays
    xs_batch = [[] for l in range(len(loader))]
    indice_batch = [[] for l in range(len(loader))]
    batch_batch = [[] for l in range(len(loader))]
    ys_batch = [[] for l in range(len(loader))]

    for ind, load in enumerate(loader): 
        generator.set_state(init)
        for j, data in enumerate(load):
            data = data.to("cuda")
            xs_batch[ind].append(data.x)
            indice_batch[ind].append(data.edge_index)
            batch_batch[ind].append(data.batch)
            ys_batch[ind].append(data.y)  
    
    for epoch in range(epochs + 1):
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0
        #Second pass: process the data 
        generator.set_state(init)
        for index, data in enumerate(loader[0]):

            optimizer.zero_grad()
            data = data.to("cuda")
            data_x = [xs_batch[i][index] for i in range(len(loader))]
            data_i = [indice_batch[i][index] for i in range(len(loader))]
            data_b = [batch_batch[i][index] for i in range(len(loader))]
            data_y =  [ys_batch[i][index] for i in range(len(loader))]

            h, out = model(data_x, data_i, data_b, train=True)
            #First data batch with Y has to have the right outputs
            loss = criterion(out, data_y[0])
            total_loss += loss / len(loader[0])
            out = F.log_softmax(out, dim=1)
            acc += accuracy(out.argmax(dim=1), data_y[0]) / len(loader[0])
            train_accs.append(acc)
            loss.backward()
            optimizer.step()

        # Validation
        generator.set_state(init)
        val_loss, val_acc = test(model, val_loader, generator, train = True, validation=True)
        val_accs.append(val_acc)

        # Print metrics every 10 epochs
        if (epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
                f'| Train Acc: {acc * 100:>5.2f}% '
                f'| Val Loss: {val_loss:.2f} '
                f'| Val Acc: {val_acc * 100:.2f}%')

    if test_loader != None:
        generator.set_state(init)
        test_loss, test_acc = test(model, test_loader, generator, validation=True)
        print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')
        test_accs.append(test_acc)
    #return model
    #print("returned lens: ", len(embeddings[0]), len(losses), len(accuracies), len(outputs), len(hs))
    return model, train_accs, val_accs, test_accs

def test(model, loaders, generator, train = False, validation = False, x_b = None, i_b = None, b_b = None):
    init = generator.get_state()
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    #First pass, append all the data together into arrays
    xs_batch = [[] for l in range(len(loaders))]
    indice_batch = [[] for l in range(len(loaders))]
    batch_batch = [[] for l in range(len(loaders))]
    ys_batch = [[] for l in range(len(loaders))]

    for i, load in enumerate(loaders): 
        generator.set_state(init)
        for j, data in enumerate(load):
            data = data.to("cuda")
            xs_batch[i].append(data.x)
            indice_batch[i].append(data.edge_index)
            batch_batch[i].append(data.batch)
            ys_batch[i].append(data.y)

    #if validation:
    #    print("validation batch: ", len(xs_batch[0]))
    #else:
    #    print("test batch: ", len(xs_batch[0]))

    #Second pass: process the data 
    generator.set_state(init)
    with torch.no_grad():
        for index, data in enumerate(loaders[0]):

            data_x = [xs_batch[i][index] for i in range(len(loaders))]
            data_i = [indice_batch[i][index] for i in range(len(loaders))]
            data_b = [batch_batch[i][index] for i in range(len(loaders))]
            data_y = [ys_batch[i][index] for i in range(len(loaders))]

            _, out = model(data_x, data_i, data_b, train)
            loss += criterion(out, data_y[0]) / len(loaders[0])            
            out = F.log_softmax(out, dim=1)
            acc += accuracy(out.argmax(dim=1), data_y[0]) / len(loaders[0])

    #if validation == False:
    #    print("test values: ", out, data_y[0])

    return loss, acc
