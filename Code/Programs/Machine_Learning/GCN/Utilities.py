'''
This file contains all the utility methods related to GCN and machine learning functions
'''
#imports
import copy
import time
import torch
torch.manual_seed(42)
from torch.profiler import profile, record_function
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
                 epochs = 100, make_loaders = False, device = 'cuda', gen_data = False):
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
    model = m_model# copy.deepcopy(m_model)
    model = model.to(device)
    for fold in range(k_fold):
        if fold >= k_fold - 1:
            gen_data = True
        else:
            gen_data = False
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
                return train_loaders, val_loaders, test_loaders, None
        #reset the model, train and save the results to the result lists
        print("RESET MODEL")
        #model = copy.deepcopy(m_model)
        #model = model.to(device)
        model, accuracies, vals, tests, all_y, all_pred, embed_data, ys = train(model, train_loaders, val_loaders, test_loaders, G, epochs, device, gen_data)
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
    print("ys coming out: ",  len(embed_data), len(ys), len(ys[0]), len(ys[0][0]))
    embed_data = convert_embed_to_reg_data(embed_data, ys, batch)
    return model, train_score, val_score, test_score, embed_data

def train(model, loader, val_loader, test_loader, generator, epochs, device, gen_data):
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
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.01,
                                weight_decay=0.001)
    model.train()
    train_accs = []
    val_accs = []
    test_accs = []
    all_pred = []
    all_y = []
    embed_data = []
    ys = []
    gen_test = False
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
            #with profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            #            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            #            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs')) as prof:
            #    for i in range(10):
            #        with record_function("model_inference"):
            #            out, embedding = model(data_x, data_i, data_b, train=True)
            #        prof.step()

            # Access the profiler results
            #print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            out, embedding = model(data_x, data_i, data_b, train=True)


            if gen_data and epoch >= epochs:
                print("IN HERE: ", epoch)
                gen_test = True
                #embed_data.append(torch.sigmoid(out))
                #ys.append(data_y)
            #stop = 5/0
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
        val_loss, val_acc, _, _ = test(model, val_loader, generator, validation=True, device = device, gen_test = gen_test, embed_data=embed_data, ys=ys)
        val_accs.append(val_acc)

        # Print metrics every 10 epochs
        if (epoch % 5 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
                f'| Train Acc: {acc * 100:>5.2f}% '
                f'| Val Loss: {val_loss:.2f} '
                f'| Val Acc: {val_acc * 100:.2f}%')

            if test_loader != None:
                generator.set_state(init)
                test_loss, test_acc, pred_y, lab_y = test(model, test_loader, generator, validation=False, device = device, gen_test = gen_test, embed_data=embed_data, ys=ys)
                all_pred += pred_y
                all_y += lab_y
                print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')
                test_accs.append(test_acc)
        #Tidy up to save memory
        del total_loss, acc, val_loss, val_acc
        #print("REMOVE BREAK FOR EPOCHS IN FUTURE")
        #break

    if test_loader != None:
        generator.set_state(init)
        test_loss, test_acc, pred_y, lab_y = test(model, test_loader, generator, validation=False, device = device, gen_test=gen_test, embed_data=embed_data, ys=ys)
        all_pred += pred_y
        all_y += lab_y
        print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')
        test_accs.append(test_acc)

    #return model
    return model, train_accs, val_accs, test_accs, all_y, all_pred, embed_data, ys

    
def test(model, loaders, generator, validation, train = False, device = 'cuda', gen_test = False, embed_data = None, ys = None):
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

            out, embedding = model(data_x, data_i, data_b, train)
            if gen_test:
                gen_test = True
                embed_data.append(embedding)
                print("whats out", out, out.shape)
                print("whats y: ", data_y, data_y[0])
                ys.append(data_y)

            loss = criterion(out, data_y[0]) / len(loaders[0]) 
            total_loss = total_loss + loss

            acc = acc + accuracy(out.argmax(dim=1), data_y[0]) / len(loaders[0])
            #Record all the predictions and labels for each fold of each test
            if validation == False:
                all_pred += out.argmax(dim=1).cpu()
                all_y += data_y[0].cpu()

    return total_loss, acc, all_pred, all_y

def load_whole_dataset(folder_names, file_name, col_names = Utilities.colnames_midhip):
    '''
    Load one or a series of datasets

    Arguments
    ---------
    folder_names: str
        file paths to the folders
    file_name: str
        file name in each folder to load
    col_names: List(Tuple()) (optional, default = colnames_midhip)
        column names to append
    
    Returns
    -------
    List(List())
        returns the loaded datasets in a list
    '''
    data = []
    for name in folder_names:
        print("loading: ", "./Code/Datasets/Joint_Data/" + str(name) + '/' + str(file_name) + "/raw/"+ str(file_name) + ".csv")
        abs_joint_data, _ = Utilities.process_data_input("./Code/Datasets/Joint_Data/" + str(name) + str(file_name) + "/raw/"+ str(file_name) + ".csv", None,
                                                                    cols=col_names, ignore_depth=False)
        data.append(abs_joint_data)
    return data


#Stitch up datasets of individuals to create a full dataset
def stitch_dataset(folder_names, stream = 1):
    '''
    stitch together multiple single person datasets into one multi-person dataset

    Arguments
    ---------
    folder_names: str
        file paths to the folders
    stream: int (optional, default = 1)
        denotes which data file type to stitch together
    
    Returns
    -------
    List(List())
        returns the loaded datasets in a list
    '''
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

    #load in the datasets 
    datasets = load_whole_dataset(folder_names, file_name)
    whole_dataset = datasets[0]
    current_instance = whole_dataset[-1][0]
    #gradually append the datasets one at a time together, saving at each step for ablation studies
    for i, dataset in enumerate(datasets):
        if i > 0:
            current_instance, whole_dataset = Utilities.assign_person_number(whole_dataset, dataset, 
                                                                       "./Code/Datasets/Joint_Data/Big/no_Sub_" + str(stream) + "_stream/" + str(i + 1) + "_people",
                                                                       i, current_instance)
    print("completed.")

def get_gait_segments(joint_data):
    '''
    Splits all data into segments of 7 for approximate gait segments

    Arguments
    ---------
    joint_data: List(List())
    
    Returns
    -------
    List(List())
        returns the dataset in the form of gait segments
    '''
    instances = []
    instance = []
    current_instance = joint_data[0][0]
    #First separate the joints into individual instances
    for joints in joint_data:
        #Append joints as normal
        if joints[0] == current_instance:
            instance.append(joints)
        else:
            #Only add certain persons examples
            instances.append(copy.deepcopy(instance))
            instance = []
            instance.append(joints)
            current_instance = joints[0]
    #Add the last instance hanging off missed by the loop
    instances.append(copy.deepcopy(instance))

    division_factor = 6
    segments = []
    for instance in instances:
        segment = []
        for i, frame in enumerate(instance):
            if i % division_factor == 0:
                if len(segment) >0:
                    segments.append(copy.deepcopy(segment))
                    segment = []
                segment.append(frame)
            else:
                segment.append(frame)
        if len(segment) == division_factor:
            segments.append(copy.deepcopy(segment))
            segment = []
            segment.append(frame)
    return segments

def convert_embed_to_reg_data(data, ys, batch_size):
    '''
    Converts embedded feature data files into gait graphs for processing

    Arguments
    ---------
    data: List(List())
        embedded data
    ys: List(int)
        corresponding class labels
    batch_size: int
        size of torch batches to segment the data into
    
    Returns
    -------
    List(List())
        returns the embedded feature dataset in the form of gait segments
    '''
    output_data = []
    instance_count = 0
    for i, batch in enumerate(data):
        #Split tensor into 32 sections of size [108, 3]
        data_cycles = torch.chunk(batch, batch_size, dim=0)
        y_cycle = torch.chunk(ys[i][0], batch_size, dim=0)
        frames = [torch.chunk(section, 6, dim=0) for section in data_cycles]
        curr_y = 0
        for j, frame_batch in enumerate(frames):
            curr_y += 1
            for k, frame in enumerate(frame_batch):
                list_frame = frame.tolist()
                meta_data = [instance_count,k,y_cycle[j].item(), 0,0,0]
                for val in list_frame:
                    meta_data.append(val)

                output_data.append(meta_data)
            instance_count += 1
    return output_data

def convert_shoe_to_format(input_file ='./code/datasets/shoedata/DIRO_skeletons.npz'):
    '''
    Converts the shoe padding dataset into a format processable by this platform

    Arguments
    ---------
    input_file: str (optional, default: './code/datasets/shoedata/DIRO_skeletons.npz')
        location for the npz file containing the skeletal information
    
    Returns
    -------
    List(List())
        returns the shoe dataset in the standard multi-dimensional python list form used by this platform.
    '''
    loaded = np.load(input_file)
    #get skeleton data of size (n_subject, n_gait, n_frame, 25*3)
    data = loaded['data']
    #print information
    print(data.shape)
    #iterate through subjects
    instance = 0
    frames = []
    for i, subject in enumerate(data):
        print(f"subject {i} of {len(data)}")
        for j, abnormality in enumerate(subject):
            print(f"abnormality {j} of {len(subject)}")
            for k, frame in enumerate(abnormality):
                print(f"frame {k} of {len(abnormality)}")
                meta_data = [instance, k, j, 0, 0, i]
                sublists = [frame[i:i+3] for i in range(0, len(frame), 3)]
                for l, coords in enumerate(sublists):
                    #print(f"value {l} of {len(frame)}")
                    if l < 21 and l != 1 and l != 15 and l != 19:
                        meta_data.append([coords[0], coords[1], coords[2]])
                #iterate instance after every series of frames
                frames.append(meta_data)
            instance += 1

    Utilities.save_dataset(frames, './code/datasets/joint_data/shoedata/3_Absolute_Data(trimmed instances)', colnames=Utilities.colnames_midhip)