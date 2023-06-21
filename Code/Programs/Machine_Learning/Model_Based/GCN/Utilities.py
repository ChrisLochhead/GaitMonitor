import Programs.Machine_Learning.Model_Based.GCN.Render as Render
import torch
from torch_geometric.loader import DataLoader as GeoLoader
from torch.utils.data import RandomSampler
import Programs.Machine_Learning.Model_Based.GCN.GAT as gat
from torch.nn.utils.rnn import pad_sequence
import copy

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

def collate_fn(data_list):

    print("is collate even being used????")
    # Separate the graphs, labels, and edge indices
    graphs = [data.x for data in data_list]
    labels = [data.y for data in data_list]
    edge_indices = [data.edge_index for data in data_list]

    # Pad the graphs to the maximum size within the batch
    batched_graph = pad_sequence(graphs, batch_first=True)

    # Compute the cumulative number of nodes per graph
    num_nodes = [data.num_nodes for data in data_list]
    cum_num_nodes = torch.tensor([0] + num_nodes).cumsum(dim=0)

    # Offset the node indices in each graph to make them unique across the batch
    for i, data in enumerate(data_list):
        data.edge_index += cum_num_nodes[i]
        data.edge_attr += cum_num_nodes[i]
        data.x += cum_num_nodes[i]

    # Combine the edge indices into a single tensor
    edge_indices_offset = []
    for i, edge_index in enumerate(edge_indices):
        edge_indices_offset.append(edge_index + cum_num_nodes[i])

    # Combine the graphs, labels, and edge indices into a single Data object
    batched_data = torch.Data(x=[1, 3], y=torch.tensor(labels), edge_index=torch.cat(edge_indices_offset, dim=1))
    return 5

# define a cross validation function
def cross_valid(MY_model, test_dataset, criterion=None,optimizer=None,datasets=None,k_fold=3, batch = 16, inputs_size = 1, epochs = 100):
    
    train_score = []
    val_score = []
    test_score = []

    total_size = len(datasets[0])
    fraction = 1/k_fold
    seg = int(total_size * fraction)
    # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset 
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size

        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))

        train_loaders = []
        val_loaders = []
        test_loaders = []

        #Set up so identical seed is used
        torch.manual_seed(1)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())  # use the same seed as Scenario 1
        G = torch.Generator()

        #There will always be at least one dataset, use samplers made for that first dataset for all of them
        train_sample = RandomSampler(datasets[0][train_indices], generator=G)
        val_sample = RandomSampler(datasets[0][val_indices], generator=G)
        test_sample = RandomSampler(test_dataset[0], generator=G)

        init = G.get_state()
        for i, dataset in enumerate(datasets):
            train_set = dataset[train_indices]
            val_set = dataset[val_indices]
            test_set = test_dataset[i]
            #print("train set: ", train_set)
            #print("first example: ", train_set[0])
            #print("batch: ", batch)
            train_loaders.append(GeoLoader(train_set, batch_size=batch, sampler = train_sample, drop_last = True))

            #for b, d in enumerate(train_loaders[-1]):
            #    print("data example: ", d, b)

            val_loaders.append(GeoLoader(val_set, batch_size=batch, sampler = val_sample, drop_last = True))
            test_loaders.append(GeoLoader(test_set, batch_size=batch, sampler = test_sample, drop_last = True))

            #Reset the generator so every dataset gets the same sampling 
            G.set_state(init)

        G.set_state(init)

        model = copy.deepcopy(MY_model)
        model = model.to("cuda")

        #model = gat.GAT(dim_in = datasets[0].num_node_features, dim_h=128, dim_out=3)
        #model = model.to("cuda")

        #print("types: ", type(MY_model), type(model), type(model))
        #stop = 5/0
            
        model, embeddings, losses, accuracies, outputs, vals, tests = train(model, train_loaders, val_loaders, test_loaders, G, epochs)
        train_score.append(accuracies[-1])
        #val_acc = valid(res_model,criterion,optimizer,val_loader)
        val_score.append(vals[-1])
        test_score.append(tests[-1])
    
    return train_score,val_score, test_score

def train(model, loader, val_loader, test_loader, generator, epochs):
    init = generator.get_state()
    #print("types: ", type(model), type(loader), type(val_loader))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.0001,
                                weight_decay=0.00005)
    epochs = epochs
    model.train()

    # Data for animations
    embeddings = []
    losses = []
    accuracies = []
    val_accs = []
    outputs = []
    hs = []
    test_accs = []
    
    for epoch in range(epochs + 1):
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0

        #First pass, append all the data together into arrays
        xs_batch = [[] for l in range(len(loader))]
        indice_batch = [[] for l in range(len(loader))]
        batch_batch = [[] for l in range(len(loader))]
        #print("lens: ", len(xs_batch))

        for i, load in enumerate(loader): 
            generator.set_state(init)
            for j, data in enumerate(load):
                data = data.to("cuda")
                xs_batch[i].append(data.x)
                indice_batch[i].append(data.edge_index)
                batch_batch[i].append(data.batch)

        #Second pass: process the data 
        generator.set_state(init)
        for index, data in enumerate(loader[0]):

            optimizer.zero_grad()
            data = data.to("cuda")

            #h, out = model([data.x], [data.edge_index], [data.batch], train=True)

            data_x = [xs_batch[i][index] for i in range(len(loader))]
            data_i = [indice_batch[i][index] for i in range(len(loader))]
            data_b = [batch_batch[i][index] for i in range(len(loader))]
            #print("len: ", len(data_x))
            h, out = model(data_x, data_i, data_b, train=True)
            #h, out = model([xs_batch[0][index]], [indice_batch[0][index]], [batch_batch[0][index]], train=True)

            #First data batch with Y has to have the right outputs

            loss = criterion(out, data.y)
            total_loss += loss / len(loader[0])
            #print("train outputs: ", data.y)
            #print("vs predictions: ", out.argmax(dim=1))
            acc += accuracy(out.argmax(dim=1), data.y) / len(loader[0])
            loss.backward()
            optimizer.step()

            embeddings.append(h)
            losses.append(loss)
            accuracies.append(acc)
            outputs.append(out.argmax(dim=1))
            hs.append(h)



        # Validation
        val_loss, val_acc = test(model, val_loader, generator, train = False)
        val_accs.append(val_acc)

        # Print metrics every 10 epochs
        if (epoch % 5 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
                f'| Train Acc: {acc * 100:>5.2f}% '
                f'| Val Loss: {val_loss:.2f} '
                f'| Val Acc: {val_acc * 100:.2f}%')

    if test_loader != None:
        test_loss, test_acc = test  (model, test_loader, generator, validation=False)
        print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')
        test_accs.append(test_acc)
    #return model
    #print("returned lens: ", len(embeddings[0]), len(losses), len(accuracies), len(outputs), len(hs))
    return model, embeddings, losses, accuracies, outputs, val_accs, test_accs

@torch.no_grad()
def test(model, loaders, generator, train = False, validation = True):
    init = generator.get_state()
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    #First pass, append all the data together into arrays
    xs_batch = [[] for l in range(len(loaders))]
    indice_batch = [[] for l in range(len(loaders))]
    batch_batch = [[] for l in range(len(loaders))]
    #print("lens: ", len(xs_batch))

    for i, load in enumerate(loaders): 
        generator.set_state(init)
        for j, data in enumerate(load):
            data = data.to("cuda")
            #print("size here", data.x.size())
            xs_batch[i].append(data.x)
            indice_batch[i].append(data.edge_index)
            batch_batch[i].append(data.batch)

    #Second pass: process the data 
    generator.set_state(init)

    for index, data in enumerate(loaders[0]):
        data = data.to("cuda")

        data_x = [xs_batch[i][index] for i in range(len(loaders))]
        data_i = [indice_batch[i][index] for i in range(len(loaders))]
        data_b = [batch_batch[i][index] for i in range(len(loaders))]

        #_, out = model(data.x, data.edge_index, data.batch, train)
        _, out = model(data_x, data_i, data_b, train)
        loss += criterion(out, data.y) / len(loaders[0])
        acc += accuracy(out.argmax(dim=1), data.y) / len(loaders[0])

        #if validation == False:
        #    print("test acc: ", acc, data.y)
        #    print("test predictions: ", out.argmax(dim=1))


    return loss, acc
