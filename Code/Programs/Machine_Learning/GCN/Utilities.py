import torch
torch.manual_seed(42)
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader as GeoLoader
from torch.utils.data import RandomSampler, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import copy
import Programs.Data_Processing.Utilities as Utilities
import Programs.Data_Processing.Dataset_Creator as Creator
from sklearn.metrics import confusion_matrix
import time


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
def cross_valid(MY_model, test_dataset, criterion=None,optimizer=None,datasets=None,k_fold=3, batch = 16, inputs_size = 1,
                 epochs = 100, type = "GAT", make_loaders = False, device = 'cuda'):
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
        start = time.time()
        print("Fold: ", fold)
        train_loaders = []
        val_loaders = []
        test_loaders = []

        #Set up so identical seed is used
        #G = torch.Generator()
        #train_sample = RandomSampler(folded_train[0][fold], generator=G,)
        ##Restrict validation and testing batch sizes to be one batch
        #val_sample = RandomSampler(folded_val[0][fold], generator=G)
        #test_sample = RandomSampler(test_dataset[0], generator=G)

        #init = G.get_state()
        print("whats i go up to here: ", len(datasets))
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
            #Reset the generator so every dataset gets the same sampling 
            #G.set_state(init)
       # G.set_state(init)

        model = copy.deepcopy(MY_model)
        model = model.to(device)
        model, accuracies, vals, tests, all_y, all_pred = train(model, train_loaders, val_loaders, test_loaders, G, epochs, batch, device)
        #model, accuracies, vals, tests, all_y, all_pred = train_individual(model, train_loaders, val_loaders, test_loaders, G, epochs, batch, device)
        total_ys += all_y
        total_preds += all_pred
        train_score.append(accuracies[-1])
        val_score.append(vals[-1])
        test_score.append(tests[-1])
        end = time.time()
        print("time elapsed: ",fold,  end - start)

    #print("final confusion: ")
    #print(confusion_matrix(total_ys, total_preds))
    f1 = f1_score(total_ys, total_preds, average='weighted')
    print("f1 score: ", f1)
    
    return model, train_score, val_score, test_score

def train(model, loader, val_loader, test_loader, generator, epochs, batch_size, device):
    init = generator.get_state()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.1,
                                weight_decay=0.001)
    model.train()
    train_accs = []
    val_accs = []
    test_accs = []
    all_pred = []
    all_y = []
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
            data_y =  [ys_batch[i][index] for i in range(len(loader))]
            out = model(data_x, data_i, data_b, train=True)

            #print("data ys all the same: ", len(data_y), len(out))
            loss = criterion(out, data_y[0]) / len(loader[0])
            total_loss = total_loss + loss
            acc =  acc + accuracy(out.argmax(dim=1), data_y[0]) / len(loader[0])
            train_accs.append(acc)
            loss.backward()
            optimizer.step()

            del data, data_x, data_i, data_b, data_y, out

        # Validation
        generator.set_state(init)
        val_loss, val_acc, _, _ = test(model, val_loader, generator, train = True, validation=True, optimizer = optimizer, device = device)
        val_accs.append(val_acc)

        # Print metrics every 10 epochs
        if (epoch % 5 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
                f'| Train Acc: {acc * 100:>5.2f}% '
                f'| Val Loss: {val_loss:.2f} '
                f'| Val Acc: {val_acc * 100:.2f}%')

            if test_loader != None:
                generator.set_state(init)
                test_loss, test_acc, pred_y, lab_y = test(model, test_loader, generator, validation=False, device = device)
                all_pred += pred_y
                all_y += lab_y
                print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')
                test_accs.append(test_acc)
        #Tidy up to save memory
        del total_loss, acc, val_loss, val_acc 

    if test_loader != None:
        generator.set_state(init)
        test_loss, test_acc, pred_y, lab_y = test(model, test_loader, generator, validation=False, device = device)
        all_pred += pred_y
        all_y += lab_y
        print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')
        test_accs.append(test_acc)

    #return model
    return model, train_accs, val_accs, test_accs, all_y, all_pred

def test(model, loaders, generator, validation, train = False, x_b = None, i_b = None, b_b = None, optimizer = None, device = 'cuda'):
    init = generator.get_state()
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0
    all_y = []
    all_pred = []
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
            data_y = [ys_batch[i][index] for i in range(len(loaders))]

            y_classes = [0,0,0,0,0,0,0,0,0]
            for d in data_y[0]:
                y_classes[d.item()] += 1

            out = model(data_x, data_i, data_b, train)
            loss = criterion(out, data_y[0]) / len(loaders[0]) 
            total_loss = total_loss + loss

            acc = acc + accuracy(out.argmax(dim=1), data_y[0]) / len(loaders[0])
            #Record all the predictions and labels for each fold of each test
            if validation == False:
                all_pred += out.argmax(dim=1).cpu()
                all_y += data_y[0].cpu()

    return total_loss, acc, all_pred, all_y

def unfold_3s_dataset(data, joint_output):
    rel_data = []
    vel_data = []
    bones_data = []

    for frame in data:
        rel_frame = frame[0:6]
        vel_frame = frame[0:6]
        bones_frame = frame[0:6]
        for i, coord in enumerate(frame):
            if i > 5:
                rel_frame.append(coord[0:3])
                vel_frame.append(coord[3:6])
                bones_frame.append(coord[6:])
        rel_data.append(rel_frame)
        vel_data.append(vel_frame)
        bones_data.append(bones_frame)
    
    Utilities.save_dataset(rel_data, joint_output + "_rel")
    Utilities.save_dataset(vel_data, joint_output + "_vel")
    Utilities.save_dataset(bones_data, joint_output + "_bone")

def load_whole_dataset(folder_names, file_name, col_names = Utilities.colnames_midhip, override = False):
    data = []
    for name in folder_names:
        print("loading: ", "./Code/Datasets/Joint_Data/" + str(name) + '/' + str(file_name) + "/raw/"+ str(file_name) + ".csv")
        if override == False:
            abs_joint_data, _ = Utilities.process_data_input("./Code/Datasets/Joint_Data/" + str(name) + str(file_name) + "/raw/"+ str(file_name) + ".csv", None,
                                                                    cols=col_names, ignore_depth=False)
        else:
            #Override to load raw data which is in a different format
            abs_joint_data, _ = Utilities.process_data_input("./Code/Datasets/Joint_Data/" + str(name) + str(file_name) + ".csv", None,
                                                                    cols=Utilities.colnames, ignore_depth=False)
        data.append(abs_joint_data)
    return data

#Make 9 class solution 
def change_to_ensemble_classes():
    joint_data, _ = Utilities.process_data_input("./Code/Datasets/Joint_Data/big/no_subtracted/2_people/raw/2_people.csv",
                                                            None, cols=Utilities.colnames_nohead, ignore_depth=False)
    Creator.convert_person_to_type(joint_data, joint_output="./Code/Datasets/Joint_Data/big/2_people_Ensemble")

#Split data by body part region 
def create_regions_data(data, folder):
    regions_data_2 = Creator.create_2_regions_dataset(data,
                                                       joint_output="./Code/Datasets/Joint_Data/"  + str(folder)  + "/2_region", images=None)
    regions_data_5 = Creator.create_5_regions_dataset(data, 
                                                      joint_output="./Code/Datasets/Joint_Data/"  + str(folder)  + "/5_region", images=None)

#Unfold the ensemble data
def extract_ensemble_data():
    joint_data, _ = Utilities.process_data_input("./Code/Datasets/Joint_Data/big/2_people/raw/2_people.csv",
                                                            None, cols=Utilities.colnames_nohead, ignore_depth=False)
    print("data loaded")
    create_regions_data(joint_data, "big")
    print("region data completed")
    unfold_3s_dataset(joint_data, joint_output="./Code/Datasets/Joint_Data/big/2_people/raw/2_people")
    print("unfolding completed")

#Remove one individual and create their own dataset for leave one out testing
def extract_single_person(data, joint_output, person = [0,1,2]):
    single_person = []
    full_dataset = []
    for i, row in enumerate(data):
        if row[5] in person:
            single_person.append(row)
        else:
            full_dataset.append(row)

    Utilities.save_dataset(single_person, joint_output + "_single")
    Utilities.save_dataset(full_dataset, joint_output + "_full")

#Stitch up datasets of individuals to create a full dataset
def stitch_dataset(folder_names, stream = 1):
    #1s file
    if stream == 1: # rel
        file_name = '/rel_data'
    elif stream == 2: # bone
        file_name = '/bone_data'
    elif stream == 3: # vel bone
        file_name = '/comb_data'
    elif stream == 4: # vel
        file_name = '/vel_data'
    elif stream == 5: #rel vel 
        file_name = '/comb_data_rel_vel'
    elif stream == 6: #rel vel bone
        file_name = '/comb_data_rel_vel_bone'


    datasets = load_whole_dataset(folder_names, file_name)
    whole_dataset = datasets[0]
    current_instance = whole_dataset[-1][0]
    for i, dataset in enumerate(datasets):
        save = False
        if i >= len(datasets) - 1:
            save = True
        if i > 0:
            current_instance, whole_dataset = Creator.assign_person_number(whole_dataset, dataset, 
                                                                       "./Code/Datasets/Joint_Data/Big/no_Sub_" + str(stream) + "_stream/" + str(i + 1) + "_people",
                                                                       i, current_instance)
    print("completed.")

#Compress datasets by removing 0s
def reduce_dataset(data, joint_output):
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            if j > 5:
                if all(item == 0 for item in value):
                    data[i][j] = ""
                    
    Utilities.save_dataset(data, joint_output)