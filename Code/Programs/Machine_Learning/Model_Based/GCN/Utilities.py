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
        torch.manual_seed(42)
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
            print("train set: ", train_set)
            print("first example: ", train_set[0])
            print("first val: ", val_set)

            train_classes = [0,0,0]
            val_classes = [0,0,0]
            for i in range(len(val_set)):
                #print("train:", train_set[i])
                #print("val: ", val_set[i])
                train_classes[train_set[i].y.item()] += 1
                val_classes[val_set[i].y.item()] += 1
            
            print("final val and test set scores: ", train_classes, val_classes)
            print("testset: ", test_set, len(test_set), len(train_set), len(val_set))
            
            train_loaders.append(GeoLoader(train_set, batch_size=batch, sampler = train_sample, drop_last = True))
            val_loaders.append(GeoLoader(val_set, batch_size=batch, sampler = val_sample, drop_last = True))
            test_loaders.append(GeoLoader(test_set, batch_size=batch, sampler = test_sample, drop_last = True))

            #Reset the generator so every dataset gets the same sampling 
            G.set_state(init)

        #G.set_state(init)

        model = copy.deepcopy(MY_model)
        #model = MY_model
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
                                lr=0.001,
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
    
    #First pass, append all the data together into arrays
    xs_batch = [[] for l in range(len(loader))]
    indice_batch = [[] for l in range(len(loader))]
    batch_batch = [[] for l in range(len(loader))]
    ys_batch = [[] for l in range(len(loader))]

    #xs_batchl = [[] for l in range(len(val_loader))]
    #indice_batchl = [[] for l in range(len(val_loader))]
    #batch_batchl = [[] for l in range(len(val_loader))]
    #y_batchl = [[] for l in range(len(val_loader))]
    #print("lens: ", len(xs_batch))

    for ind, load in enumerate(loader): 
        generator.set_state(init)
        for j, data in enumerate(load):
            data = data.to("cuda")
            xs_batch[ind].append(data.x)
            indice_batch[ind].append(data.edge_index)
            batch_batch[ind].append(data.batch)
            ys_batch[ind].append(data.y)
    
    #for ind, load in enumerate(val_loader): 
    #    generator.set_state(init)
    #    for j, data in enumerate(load):
    #        data = data.to("cuda")
    #        xs_batchl[ind].append(data.x)
    #        indice_batchl[ind].append(data.edge_index)
    #        batch_batchl[ind].append(data.batch)
    #        y_batchl[ind].append(data.y)

    for epoch in range(epochs + 1):
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0
        #t_acc = 0
        #t_loss = 0
        #Second pass: process the data 
        generator.set_state(init)
        for index, data in enumerate(loader[0]):

            optimizer.zero_grad()
            data = data.to("cuda")

            #h, out = model([data.x], [data.edge_index], [data.batch], train=True)

            data_x = [xs_batch[i][index] for i in range(len(loader))]
            data_i = [indice_batch[i][index] for i in range(len(loader))]
            data_b = [batch_batch[i][index] for i in range(len(loader))]
            data_y =  [ys_batch[i][index] for i in range(len(loader))]

            #data_lx = [xs_batchl[i][0] for i in range(len(val_loader))]
            #data_li = [indice_batchl[i][0] for i in range(len(val_loader))]
            #data_lb = [batch_batchl[i][0] for i in range(len(val_loader))]
            #data_ly =  [y_batchl[i][0] for i in range(len(val_loader))]

            #print("lens: ", len(val_loader[0]), len(loader[0]))
        
            #print("len: ", len(data_x))
            h, out = model(data_x, data_i, data_b, train=True)
            #h, out = model([xs_batch[0][index]], [indice_batch[0][index]], [batch_batch[0][index]], train=True)
            #First data batch with Y has to have the right outputs

            loss = criterion(out, data_y[0])
            total_loss += loss / len(loader[0])
            #print("train outputs: ", data.y)
            #print("vs predictions: ", out.argmax(dim=1))
            acc += accuracy(out.argmax(dim=1), data_y[0]) / len(loader[0])
            #t_a, t_l = single_test(model, data_lx, data_li, data_lb, data_ly, loader)
            #t_acc += t_a / len(loader[0])
            #t_loss += t_l / len(loader[0])
            loss.backward()
            optimizer.step()

            embeddings.append(h)
            losses.append(loss)
            accuracies.append(acc)
            outputs.append(out.argmax(dim=1))
            hs.append(h)



        # Validation
        val_loss, val_acc = test(model, val_loader, generator, train = True, validation=True)
        val_accs.append(val_acc)

        # Print metrics every 10 epochs
        if (epoch % 5 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
                f'| Train Acc: {acc * 100:>5.2f}% '
                f'| Val Loss: {val_loss:.2f} '
                f'| Val Acc: {val_acc * 100:.2f}%')

    if test_loader != None:

        test_loss, test_acc = test(model, test_loader, generator, validation=True)
        print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')
        test_accs.append(test_acc)
    #return model
    #print("returned lens: ", len(embeddings[0]), len(losses), len(accuracies), len(outputs), len(hs))
    return model, embeddings, losses, accuracies, outputs, val_accs, test_accs


def single_test(model, data_x, data_i, data_b, data_y, loaders):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    _, out = model(data_x, data_i, data_b, train = False)
    loss = criterion(out, data_y[0]) #/ len(loaders[0])
    acc = accuracy(out.argmax(dim=1), data_y[0]) #/ len(loaders[0])

    #print("accuracy hereee: ", acc)
    return acc, loss

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
    #print("lens: ", len(xs_batch))

    for i, load in enumerate(loaders): 
        generator.set_state(init)
        for j, data in enumerate(load):
            data = data.to("cuda")
            #print("size here", data.x.size())
            xs_batch[i].append(data.x)
            indice_batch[i].append(data.edge_index)
            batch_batch[i].append(data.batch)
            ys_batch[i].append(data.y)

    #Second pass: process the data 
    generator.set_state(init)
    with torch.no_grad():
        for index, data in enumerate(loaders[0]):
            #data = data.to("cuda")

            data_x = [xs_batch[i][index] for i in range(len(loaders))]
            data_i = [indice_batch[i][index] for i in range(len(loaders))]
            data_b = [batch_batch[i][index] for i in range(len(loaders))]
            data_y = [ys_batch[i][index] for i in range(len(loaders))]

            #print("same? " ,data.y)
            #print("same? ", data_y)

            #_, out = model(data.x, data.edge_index, data.batch, train)
            #print("validation data: ", data_x)
            #print("length: ", data_x[0].size())
            #print("y value: ", data_y)
            #print("batches: ", data_b, data_b[0].size())

            #batch_indices = data_b[0].tolist()
            #indice_now = 0
            #indice_counts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

            #for indice in batch_indices:
            #    if indice_now != indice:
            ##        indice_now = indice
            #    else:
            #        indice_counts[indice] += 1

            #print("batch values:", indice_counts)
            #my_data = data_x[0].tolist()
            #iter_count = 0
            #current_batch = indice_counts[0]
            #current_batch_indice = 0
            #for i, d in enumerate(my_data):
                #Display first of each batch
            #    if iter_count == 1:
            #        print("display data: ", d)
            #        print("batch: ", current_batch_indice)
            #        print("corresponding y: ", data_y)
                
            #    iter_count += 1

            #    if iter_count >= current_batch:
            #        iter_count = 0
            #        current_batch_indice += 1
            #        if current_batch_indice < len(indice_counts):
            #            current_batch = indice_counts[current_batch_indice]
                            

            #print("total: ", sum(indice_counts), len(data_x[0]))


            #stop = 5/0
            _, out = model(data_x, data_i, data_b, train)
            loss += criterion(out, data_y[0]) / len(loaders[0])
            acc += accuracy(out.argmax(dim=1), data_y[0]) / len(loaders[0])

        #if validation == True:
        #    print("test acc: ", acc)# data.y)
                    #print("test predictions: ", out.argmax(dim=1))


    return loss, acc
