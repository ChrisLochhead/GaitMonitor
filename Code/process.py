'''
This is the central file for running all dataset processing and machine learning routines.
'''
#dependencies
from Programs.Data_Processing.Image_Processor import *
import Programs.Data_Processing.Dataset_Creator as Creator
import Programs.Machine_Learning.GCN.Dataset_Obj as Dataset_Obj
import Programs.Data_Processing.Render as Render
import Programs.Machine_Learning.GCN.GAT as gat
import Programs.Machine_Learning.GCN.GCN_GraphNetwork as stgcn
import Programs.Machine_Learning.GCN.Utilities as graph_utils
import Programs.Machine_Learning.GCN.vae_utils as vae_utils
from Programs.Data_Processing.kmeans_interp.kmeans_feature_imp import KMeansInterp

#imports
import time
import random
import math
import pandas as pd
random.seed(42)
import torch
torch.manual_seed(42)
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def process_images_into_joints(folder):
    '''
    This is a sub function containing the first part of the pre-processing pipeline, namely involving the computationally expensive joint-position extraction
    from raw images. 

    Arguments
    ---------
    folder : string
        Folder name for where to find the images and store the outputs. 
    
    Returns
    -------
    List(List), List(List)
        Returns 2 2D lists: one containing the joint information, the second returning the images in the same order.

    See Also
    --------
    This has its own function as, unlike the variable normalization functions which need to be experimented with in ablation studies, it should only be ran once per dataset.
    '''
    #Extract joints from images
    run_images("./Code/Datasets/WeightGait/Full_Dataset/", out_folder="./Code/Datasets/Joint_Data/" + str(folder) + "/", exclude_2D=False, 
              start_point=0)
    #Remove empty frames
    abs_joint_data, image_data = Creator.process_empty_frames(joint_file="./Code/Datasets/Joint_Data/" + str(folder) + "/Absolute_Data.csv",
                                                 image_file="./Code/Datasets/Individuals/" + str(folder) + "/Full_Dataset/",
                                                 joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/2_Absolute_Data(empty frames removed)",
                                                  image_output="./Code/Datasets/Individuals/" + str(folder) + "/2_Empty Frames Removed/")
    #Trim start and end frames where joints get confused by image borders
    abs_joint_data, image_data =Creator.process_trimmed_frames(abs_joint_data, image_data,
                                                        joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/3_Absolute_Data(trimmed instances)",
                                                         image_output="./Code/Datasets/" + str(folder) + "/3_Trimmed Instances/", trim = 5, include_joints=True)
    return abs_joint_data, image_data


def process_joint_normalization(folder, abs_joint_data, image_data, scale_joints, norm_joints):
    '''
    This function concerns the functions in the pre-processing pipeline that implement joint scaling and normalization, in an attempt to reduce the individuality across people, without
    sacrificing the discriminatory information provided by the gait abnormalities. It also serves to reduce the scales of the values for easier processing in a neural network.

    Arguments
    ---------
    folder : string
        Folder name for where to find the images and store the outputs. 
    
    abs_joint_data : List(List)
        2D list of joint data

    image_data : List(List)
        2D list of image data, corresponding to abs_joint_data

    Returns
    -------
    List(List), List(List)
        Returns 2 2D lists: one containing the joint information, the second returning the images in the same order.

    See Also
    --------
    This function is not called during normalization ablations studies.
    ''' 
    print("lens: ", len(image_data), len(abs_joint_data))
    if norm_joints:
        #Render.render_joints_series(image_data, abs_joint_data, 10, True)
        abs_joint_data = Creator.create_normalized_dataset(abs_joint_data, image_data, joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/5_Absolute_Data(norm)")
    if scale_joints:
        abs_joint_data = Creator.create_scaled_dataset(abs_joint_data, image_data, joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/5_Absolute_Data(scaled)")
    return abs_joint_data, image_data

def generate_single_stream_dataset(folder, joint_data, prefix = 'rel', subtr = True):
    '''
    This generates a single stream dataset of either relative, joint or bone data by passing them through the skeleton-subtraction methodology for data cleaning and generates
    dummy values using isotropic gaussian noise.

    Arguments
    ---------
    folder : string
        Folder name for where to find the images and store the outputs. 
    
    joint_data : List(List)
        2D list of joint co-ordinates
    
    prefix : string
        either rel, vel, or bone to denote which type of joint co-ordinates are being processed.
    
    Returns
    -------
    List(List), List(List)
        Returns the subtracted and dummied datasets

    '''
    if subtr:
        joint_data = Creator.subtract_skeleton(joint_data, 
                                                        joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/" + str(prefix) + "_Subtracted",
                                                        base_output="./Code/Datasets/Joint_Data/" + str(folder) + "/" + str(prefix) +"_base")

    dum_data = Creator.create_dummy_dataset(joint_data, 
                                                 joint_output=         "./Code/Datasets/Joint_Data/" + str(folder) + "/" + str(prefix) +"_data")
    
    return joint_data, dum_data

def process_data(folder = "Chris", run_ims = False, norm_joints = True, scale_joints = True, subtract = True):
    '''
    This function calls the pre-processing pipeline that takes raw images and extracts the joint positions, then turns them into various representations and 
    datasets including relative position, velocity, joint angles among other pre-processing steps.

    Arguments
    ---------
    folder : string
        Folder name for where to find the images and store the outputs. 
    
    Returns
    -------
    None

    See Also
    --------
    Datasets are saved throughout the course of this function, so it hasn't been designed to require a return value.
    '''
    #Either run the images to get the joints the first time or pre-load them in
    if run_ims:
        abs_joint_data, image_data = process_images_into_joints(folder)
    else:
        abs_joint_data, image_data = Utilities.process_data_input("./Code/Datasets/Joint_Data/" + str(folder) + "/3_Absolute_Data(trimmed instances)/raw/3_Absolute_Data(trimmed instances).csv",
                                                              "./Code/Datasets/Individuals/" + str(folder) + "/3_Trimmed Instances/", cols=Utilities.colnames, ignore_depth=False)

    #Add the mid-hip joint
    abs_joint_data = Creator.append_midhip(abs_joint_data, image_data, 
                                                   joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/5_Absolute_Data(midhip)")

    #Save the un-normalized absolute joints for calculating bone angles later
    imperfect_joints = copy.deepcopy(abs_joint_data)

    abs_joint_data, image_data = process_joint_normalization(folder, abs_joint_data, image_data, scale_joints, norm_joints)
    
    #Create relative, velocity and bone datasets
    print("\nStage 7: Relativizing data")
    relative_joint_data = Creator.create_relative_dataset(abs_joint_data, image_data,
                                                 joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/Relative_Data")
    velocity_data = Creator.create_velocity_dataset(abs_joint_data, image_data, 
                                                joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/Velocity_Data")
    joint_bones_data = Creator.create_bone_dataset(imperfect_joints, image_data, 
                                                   joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/Bones_Data")
    
    #Apply subtraction and dummying
    relative_joint_data, rel_dum_data = generate_single_stream_dataset(folder, relative_joint_data, prefix='rel', subtr=subtract)
    velocity_data, vel_dum_data = generate_single_stream_dataset(folder, velocity_data, prefix='vel', subtr=subtract)
    joint_bones_data, bone_dum_data = generate_single_stream_dataset(folder, joint_bones_data, prefix='bone', subtr=subtract)

    #Combine datasets
    #Creator.combine_datasets(rel_dum_data, vel_dum_data, None, image_data,
    #                                        joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/comb_data_rel_vel")
    #Creator.combine_datasets(rel_dum_data, vel_dum_data, bone_dum_data, image_data,
    #                                        joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/comb_data_rel_vel_bone")
    #Creator.combine_datasets(vel_dum_data, bone_dum_data, None, image_data,
    #                                         joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/comb_data_bone_vel")


def load_datasets(types, folder, multi_dim = False, class_loc = 2, num_classes = 9):
    '''
    This loads in datasets selected datasets for training.

    Arguments
    ---------
    types: List(int)
        
    
    folder : string
        Folder name for where to find the images and store the outputs. 
    
    Returns
    -------
    None

    See Also
    --------
    Datasets are saved throughout the course of this function, so it hasn't been designed to require a return value.
    '''
    datasets = []
    print("loading datasets...")
        
    for i, t in enumerate(types):
        print("loading dataset {} of {}. ".format(i + 1, len(types)), t)
        #Type 1: Normal, full dataset
        if t == 1:  
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/rel_data', 
            #                            'rel_data.csv',             
            #                                joint_connections=Render.joint_connections_n_head))   
            datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/2_people', 
                            '2_people.csv',             
                                joint_connections=Render.bottom_joint_connection, class_loc=class_loc, num_classes=num_classes))   
            #2s ST-GCN
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + "no_sub_1_stream" + f'/{num_people}_people', 
            #                            f'{num_people}_people.csv',             
            #                                joint_connections=Render.joint_connections_n_head))      
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + "no_sub_4_stream"+ f'/{num_people}_people', 
            #                            f'{num_people}_people.csv',             
            #                                joint_connections=Render.joint_connections_n_head))     
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + "no_sub_2_stream"+ f'/{num_people}_people', 
            #                            f'{num_people}_people.csv',             
            #                                joint_connections=Render.joint_connections_n_head))     

            #9d
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + "/multi_dim_1_13/", 
            #                'multi_dim_1_13.csv',             
            #                    joint_connections=Render.joint_connections_n_head))   
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + "/multi_dim_2_13/", 
            #    'multi_dim_2_13.csv',             
            #        joint_connections=Render.joint_connections_n_head))   
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + "/multi_dim_3_13/", 
            #    'multi_dim_3_13.csv',             
            #        joint_connections=Render.joint_connections_n_head))   
        #Type 2: HCF dataset (unused)
        elif t == 2:
            #This MUST have cycles, there's no non-cycles option
            dataset = Dataset_Obj.HCFDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/13.5_HCF_Data(normed)',
                                                    '13.5_HCF_Data(normed).csv', cycles=True)
            datasets.append(dataset)

    print("datasets loaded.")
    #This function converts the classes from 1-3 to 1-9, where each class is a mixture of the weight and walking type. 
    #It stores this in the "person" feature in the metadata for convienience.
    if multi_dim:
        for index, dataset in enumerate(datasets):
            datasets[index] = Creator.convert_person_to_type(dataset, None)
    #Return requested datasets
    return datasets

def process_datasets(datasets, leave_one_out = False):
    '''
    Translates the loaded in datasets into train, validation and test sets, applying any other bespoke conditions like leave-one-out

    Arguments
    ---------
    datasets: List(Datasets)
        List of datasets
    leave_one_out: bool (optional, default = True)
        Indicates whether to split the train and test in a leave-one-out fashion

    Returns
    -------
    List(List()), List(List())
        Returns the train and test split datasets

    '''
    print("Processing data...")
    train_indice_list = []
    test_indice_list = []
    dataset_size = len(datasets[0])

    #Uncomment this for standard train test split
    train_indices = random.sample(range(dataset_size), int(0.8 * dataset_size)) # check if some duplicates appear??????
    test_indices = random.sample(set(range(dataset_size)) - set(train_indices), int(0.2 * dataset_size)) # Just take from dataset without 
    print("start lens: ", len(train_indices), len(test_indices), len(train_indices) + len(test_indices))
    for dataset in datasets:
        if leave_one_out:
            #Logic needs to be manually changed here for whatever effect you want: the default is moving all instances of person 0 into the test set
            for i, data in enumerate(dataset):
                if data.person == 0:
                    if i in train_indices:
                        test_indices.append(i)
                        train_indices.remove(i)
                if data.person != 0:
                    if i in test_indices:
                        test_indices.remove(i)
                        train_indices.append(i)
    
        print("final lens: ", len(train_indices), len(test_indices), len(train_indices) + len(test_indices))
        train_indice_list.append(train_indices)
        test_indice_list.append(test_indices)

    #These regions will be the same for both datasets
    multi_input_train_val = []
    multi_input_test = []
    for i, dataset in enumerate(datasets):
        multi_input_train_val.append(dataset[train_indice_list[i]])
        multi_input_test.append(dataset[test_indice_list[i]])

    return multi_input_train_val, multi_input_test

def process_results(train_scores, val_scores, test_scores):
    '''
    Calculates and prints various result metrics from the experiment

    Arguments
    ---------
    train_scores, val_scores, test_scores: List()
        Lists of the various accuracy scores
        
    Returns
    -------
    None

    '''
    print("Final results: ")
    for ind, t in enumerate(test_scores):
        test_scores[ind] = test_scores[ind].cpu()
        test_scores[ind] = float(test_scores[ind])

    for i, score in enumerate(train_scores):
        print("score {:.2f}: training: {:.2f}%, validation: {:.2f}%, test: {:.2f}%".format(i, score * 100, val_scores[i] * 100, test_scores[i] * 100))

    mean, var = Utilities.mean_var(test_scores)
    print("mean, std and variance: {:.2f}%, {:.2f}% {:.5f}".format(mean, math.sqrt(var), var))

def run_model(dataset_types, hcf, batch_size, epochs, folder, save = None, load = None, leave_one_out = False, dim_out = 3, class_loc = 2, model_type = 'VAE', vae=False,
              save_embedding = False, embedding_size = 3, gen_data = False):
    '''
    Runs the model configuration

    Arguments
    ---------
    dataset_types: List(int)
        list of ints that denote which datasets will be loaded
    hcf: bool
        indicates whether to use hcf data
    batch_size: int
        batch size
    epochs: int
        epochs
    folder: str
        name of the root folder for the datasets
    save: str (optional, default = None)
        indicates whether to save the model weights
    load: str (optional, default = None)
        indicates whether to load model weights 
    leave_one_out: bool (optional, default = False)
        indicates whether to use a leave-one-out test configuration
    multi_dim: bool (optional, default = False)
        indicates whehter to use 3 or 9-class problem space
    num_people: int (optional, default = 3)
        indicates the size of the dataset to load by people

    Returns
    -------
    None

    '''
    #Load the full dataset
    datasets = load_datasets(dataset_types, folder, False, class_loc, dim_out)
    print("datasets here: ", datasets)
    #Concatenate data dimensions for ST-GCN
    data_dims = []
    for dataset in datasets:
        data_pair = [dataset.num_features, dataset.num_node_features]
        data_dims.append(data_pair)

    num_datasets = len(datasets)
    print("dataset info: ", len(datasets[0]), dataset[0])
    #Accounting for extra original dataset not used in dummy case for training but only for testing
    if 5 in dataset_types:
        num_datasets -= 1

    print("number of datasets: ", num_datasets)
    multi_input_train_val, multi_input_test = process_datasets(datasets, leave_one_out=leave_one_out)
    datasets = [datasets[0]]
    num_datasets = len(datasets)

    print("\nCreating {} datasets: ".format(len(datasets)))
    print("dataset info:", datasets[0].max_cycle, datasets[0].num_nodes_per_graph)
    print("going in: ", datasets[0].num_node_features)
    model = stgcn.GCN_GraphNetwork(dim_in=[d.num_node_features for d in datasets], dim_h=32, num_classes=dim_out, n_inputs=num_datasets,
                                data_dims=data_dims, batch_size=batch_size, hcf=hcf,
                                max_cycle=datasets[0].max_cycle, num_nodes_per_graph=datasets[0].num_nodes_per_graph, device = device, type=model_type)
    
    if load != None:
        print("loading model")
        #model.load_state_dict(torch.load('./Code/Datasets/Weights/' + str(load) + '.pth'),  strict=False)
        # Load weights from the first network into the second network

        #CHANGE THIS TO FIX LOADING
        first_weights = torch.load('./Code/Datasets/Weights/' + str(load) + '.pth')
        second_weights = model.state_dict()

        # Transfer weights except for the last layer
        print("how many: ", len(first_weights.items()))
        for i, (name, param) in enumerate(first_weights.items()):
            print(f"Name {name} and layer number: {i}")
            if i >= len(first_weights.items()) - 2:
                print(f"name {name} not in second weights in number {i}")
                continue
            second_weights[name].copy_(param)

        model.load_state_dict(second_weights)

    model = model.to(device)
    if vae == False:
        model, train_scores, val_scores, test_scores, embed_data = graph_utils.cross_valid(model, multi_input_test, datasets=multi_input_train_val,
                                                                     k_fold=2, batch=batch_size, epochs=epochs, device = device, gen_data = gen_data)
    else:
        model, train_scores, val_scores, test_scores, embed_data = vae_utils.cross_valid(model, multi_input_test, datasets=multi_input_train_val,
                                                                k_fold=2, batch=batch_size, epochs=epochs, device = device)
        
    if save_embedding:
        print("saving outputs ")
        print("embed data 1: ", len(embed_data), len(embed_data[0]))
        Utilities.save_dataset(embed_data, './code/datasets/joint_data/embed_data/2_people_1')
        embed_data = sorted(embed_data, key=lambda x: x[2])
        print("embed data 2: ", len(embed_data), len(embed_data[0]))
        Utilities.save_dataset(embed_data, './code/datasets/joint_data/embed_data/2_people_2')
        embed_data = reorder_instance_numbers(embed_data)
        print("embed data 3: ", len(embed_data), len(embed_data[0]))
        Utilities.save_dataset(embed_data, './code/datasets/joint_data/embed_data/2_people_3')
        embed_data = extract_embed_data(embed_data)
        print("embed data 4: ", len(embed_data), len(embed_data[0]))
        Utilities.save_dataset(embed_data, './code/datasets/joint_data/embed_data/2_people_4')

    if save != None:
        print("saving model")
        torch.save(model.state_dict(), './Code/Datasets/Weights/' + str(save) + '.pth')
    #Process and display results
    process_results(train_scores, val_scores, test_scores)

def convert_directory_to_videos(parent_folder, output, depth = False):
    '''
    creates a series of videos from source images

    Arguments
    ---------
    parent_folder: str
        location of root folder for the images
    output: str
        output file location
    depth: bool (optional, default = False)
        indicates whether to include depth images
        
    Returns
    -------
    None
    '''
    image_data = []
    directory = os.fsencode(parent_folder)
    for subdir_iter, (subdir, dirs, files) in enumerate(os.walk(directory)):
        dirs.sort(key=Utilities.numericalSort)
        split = str.split(subdir.decode('utf-8'), '/')

        #avoid empty leading folders
        print("split: ", split)
        if split[-1] != "":
            print("converting to vid")
            Utilities.convert_to_video(subdir.decode('utf-8'), output + split[-1], split[-1], depth=depth)

def create_datasets(streams = [1,2,3,4,5,6]):
    '''
    creates a series of datasets from individual person datasets

    Arguments
    ---------
    streams: List(int)
        number of streams to stitch together
        
    Returns
    -------
    None
    '''
        #Assign person numbers and uniform instance counts:
    folder_names = ['ahmed', 'Amy', 'Anna', 'bob', 'cade', 'emma', 'erin', 
                    'grant', 'pheobe', 'scarlett', 'sean c', 'sean g', 'wanok']
    for p in folder_names:
        #Current run: 1, 0, 1
        process_data(p, run_ims=False, norm_joints=True, scale_joints=True, subtract=True)
    
    for s in streams:
        graph_utils.stitch_dataset(folder_names=folder_names, stream=s)

def make_comb(folder, rel_path, vel_path, bone_path):
    '''
    creates a series of datasets from individual person datasets

    Arguments
    ---------
    folder: str
        source folder location
    rel_path: str
        location of the relative data
    vel_path: str
        location of velocity data
    bone_path: str
        location of bone angle data
        
    Returns
    -------
    None
    '''
    rel, _ = Utilities.process_data_input(rel_path, None)
    vel, _ = Utilities.process_data_input(vel_path, None)
    bone, _ = Utilities.process_data_input(bone_path, None)

    Creator.combine_datasets(rel, vel, bone, None,
                                                joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/comb_data_rel_vel_bone")
    
    Creator.combine_datasets(rel, vel, None, None,
                                                joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/comb_data_rel_vel")
    

#TEMP FUNCTIONS HERE
########################################################################################################################################################################################### 
def reorder_instance_numbers(data):
    curr = -1
    instance_no = 0
    for i, row in enumerate(data):
        if row[0] != curr:
            if curr != -1:
                instance_no += 1
            curr = row[0]
        data[i][0] = instance_no

    return data

def extract_embed_data(data):
    new_data = []
    for i, row in enumerate(data):
        new_row = []
        for j, val in enumerate(row):
            if j <= 5: 
                new_row.append(val)
            else:
                for embed in val:
                    new_row.append(embed)
        new_data.append(new_row)
    return new_data
                
            
from sklearn.preprocessing import MinMaxScaler
def apply_standard_scaler(data, output):
    data, _ = Utilities.process_data_input(data, None)
    #remove all metadata
    meta_data = [row[:6] for row in data]
    joints_data = [row[6:] for row in data]
    print("correct?, ", meta_data[0])
    print("and this: ", joints_data[0])
    #unwrap all joints
    unwrapped_joints = [[value for sublist in row for value in sublist] for row in joints_data]
    print("unwrap: ", unwrapped_joints[0], len(unwrapped_joints[0]))
    #apply scaler
    # Initialize StandardScaler
    scaler =  MinMaxScaler(feature_range=(0, 1))

    # Fit the scaler to your data (calculate mean and standard deviation)
    scaler.fit(unwrapped_joints)

    # Transform the data (apply standard scaling)
    scaled_joints = scaler.transform(unwrapped_joints)
    #rewrap all joints
    rewrapped_joints = []
    for i, row in enumerate(scaled_joints):
        coord = []
        joints_row = []
        for j, val in enumerate(row):
            if j != 0 and j % 3 == 0:
                joints_row.append(copy.deepcopy(coord))
                coord = []
            coord.append(val)
        joints_row.append(copy.deepcopy(coord))
        rewrapped_joints.append(joints_row)

    #print("rewrapped: ", rewrapped_joints[0], len(rewrapped_joints[0]))
    #stop = 5/0
    #readd metadata
    for i, row in enumerate(rewrapped_joints):
        #print("prior:", rewrapped_joints[i])
        #print("sizes? ", len(rewrapped_joints[i]))
        rewrapped_joints[i][:0] = meta_data[i]
        #print("readded:", rewrapped_joints[i])
        #print("sizes? ", len(rewrapped_joints[i]))
        #stop = 5/0

    Utilities.save_dataset(rewrapped_joints, output)

def remove_z(joint_source, joint_output):
    joints, _ = Utilities.process_data_input(joint_source, None)

    for i, row in enumerate(joints):
        for j, coord in enumerate(row):
            if j > 5:
                joints[i][j][2] = 0
    
    Utilities.save_dataset(joints, joint_output)
    #81 vs 

def replace_nans(joint_source, joint_output):
    data, _ = Utilities.process_data_input(joint_source, None)
    Utilities.save_dataset(data, joint_output)

def convert_shoe_to_format():
    '''Example of Python code reading the skeletons'''
    loaded = np.load('./code/datasets/shoedata/DIRO_skeletons.npz')

    #get skeleton data of size (n_subject, n_gait, n_frame, 25*3)
    data = loaded['data']

    #get joint coordinates of a specific skeleton
    skel = data[0,0,0,:]
    x = [skel[i] for i in range(0, len(skel), 3)]
    y = [skel[i] for i in range(1, len(skel), 3)]
    z = [skel[i] for i in range(2, len(skel), 3)]

    #get default separation
    separation = loaded['split']

    #print information
    print(data.shape)
    print(separation)
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
    #try making dataframe

    #expected results:
    #(9, 9, 1200, 75)
    #['train' 'test' 'train' 'test' 'train' 'train' 'test' 'test' 'train']

def scale_values_in_data(data):
    for i, row in enumerate(data):
        print(f"row {i} of {len(data)}")
        for j, coord in enumerate(row):
            #print("in here: ", j, coord)
            if j > 5:
                #print("before: ", data[i][j])
                data[i][j][0] *= 100
                data[i][j][1] *= 100
                data[i][j][2] *= 100
                #print("after: ", data[i][j])
        #stop = 5/0
    return data

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment
from scipy.stats import mode
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def dimensionality_reduction(data):
    # Load sample data (digits dataset)
    data = pd.DataFrame(data)
    # Assuming df is your pandas DataFrame with the first 6 columns as metadata and the last column as class
    # Remove metadata columns and keep only the features and class
    print("data: ", data)
    features = data.iloc[:, 6:].values
    labels = data.iloc[:, 2].values  # Assuming class is the 3rd column
    
    # Initialize and fit TSNE
    tsne = TSNE(n_components=3, random_state=42)  # Reduce to 2D
    X_reduced = tsne.fit_transform(features)

    # Plot the reduced data
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='tab10', alpha=0.5)
    plt.title('TSNE visualization of digits dataset')
    plt.colorbar(scatter, label='Digit Label')
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')
    plt.show()


def apply_grouped_pca(data):
   # Initialize a list to store the PCA features for each group
    pca_features = []

    # Number of groups
    num_groups = 18

    # Apply PCA to each group of 6 features
    for group_num in range(num_groups):
        # Calculate the indices for the current group
        feature_indices = [group_num + i * num_groups for i in range(6)]
        
        # Select the group of 6 features
        feature_group = data.iloc[:, feature_indices]
        
        # Initialize PCA
        pca = PCA(n_components=1)  # Assuming you want to keep 1 principal component per group
        
        # Fit PCA to the feature group and transform it
        pca_result = pca.fit_transform(feature_group)
        
        # Append the PCA feature to the list
        pca_features.append(pca_result.flatten())

    # Concatenate the PCA features into a DataFrame
    pca_df = pd.DataFrame(pca_features).T

    # Rename the columns to represent the PCA features
    pca_df.columns = [f'PCA_Feature_{group_num + 1}' for group_num in range(num_groups)]

    # Display the resulting DataFrame
    return pca_df

def k_means_experiment(data):
    data = pd.DataFrame(data)
    # Remove metadata columns and keep only the features and class
    features = data.iloc[:, 6:].values
    labels = data.iloc[:, 2].values 

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Standardize the features
    # Separate the data into labeled and unlabeled based on the class label
    '''
    labeled_indices = np.where(labels == 0)[0]
    unlabeled_indices = np.where(labels != 0)[0]

    labeled_data = features[labeled_indices]
    unlabeled_data = features[unlabeled_indices]
    labeled_data  = labeled_data.tolist()
    '''
    features = apply_grouped_pca(pd.DataFrame(features))

    # Initialize and fit KMeans model on the training data
    kmeans = KMeans(n_clusters=3)  # Assuming 3 clusters for the three classes
    kmeans.fit(features)
    cluster_labels = kmeans.labels_
    # Map clusters to the true class labels
    # For each cluster, find the most common true label
    cluster_to_class = {}
    for i in range(3):
        mask = cluster_labels == i
        cluster_to_class[i] = mode(labels[mask]).mode[0]

    # Map the cluster labels to the true class labels
    predicted_labels = np.array([cluster_to_class[cluster] for cluster in cluster_labels])
    accuracy = accuracy_score(labels, predicted_labels)

    centroids = kmeans.cluster_centers_
    distance_01 = np.linalg.norm(centroids[0] - centroids[1])  # Distance between cluster 0 and 1
    distance_02 = np.linalg.norm(centroids[0] - centroids[2])  # Distance between cluster 0 and 2
    distance_12 = np.linalg.norm(centroids[1] - centroids[2])  # Distance between cluster 1 and 2

    # Step 2: Calculate differences between centroids for each feature
    centroid_differences = np.max(centroids, axis=0) - np.min(centroids, axis=0)

    # Step 3: Calculate feature variability within each cluster
    cluster_variances = []
    for cluster in range(3):
        cluster_data = data[kmeans.labels_ == cluster]
        cluster_variance = cluster_data.var()
        cluster_variances.append(cluster_variance)

    print("cluster variances: ", type(cluster_variances), len(cluster_variances), len(cluster_variances[0]), len(cluster_variances[1]), len(cluster_variances[2]))
    # Combine the feature differences and variabilities to rank features for each cluster
    # Create a dictionary to store the importance scores
    feature_importance = {}
    for feature_index in range(18):
        #print("how big is this: ", data.shape[1])
        #feature_name = features.columns[feature_index]
        importance_scores = []
        for cluster in range(3):
            # Calculate importance as the inverse of variance within the cluster
            # times the difference between centroids
            importance = 1 / ((cluster_variances[cluster][feature_index] * centroid_differences[feature_index]) + 0.0001)

            importance_scores.append((cluster, importance))

        # Sort the importance scores for each feature
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        feature_importance[feature_index] = importance_scores

    # Number of features in each group
    group_size = 6
    group_index = 0
    current_group_importance = {}
    cluster_importances = [[],[],[]]
    # Iterate through each feature and its importance scores in different clusters
    for idx, (feature_name, importance_scores) in enumerate(feature_importance.items()):
        # Determine the current group by dividing the index by the group size
        current_group_index = idx // group_size
        
        # If we move to a new group, reset the cumulative importance dictionary and print the previous group's results
        if current_group_index > group_index:
            # Print the cumulative importance of the previous group
            #print(f'Group {group_index + 1}:')
            for cluster, importance in current_group_importance.items():
                #print(f'Cluster {cluster}: Cumulative Importance = {importance:.4f}')
                cluster_importances[cluster].append(importance)
            # Reset the cumulative importance for the new group
            current_group_importance = {}
            group_index = current_group_index
        
        # Iterate through the importance scores for each cluster
        for cluster, importance in importance_scores:
            # Add the importance to the cumulative sum for the current group
            if cluster not in current_group_importance:
                current_group_importance[cluster] = 0
            current_group_importance[cluster] += importance

    #Need to do PCA on original features 
    features.columns = ['Nose','L_eye','R_eye','L_ear','R_ear','L_shoulder','R_shoulder',
    'L_elbow','R_elbow','L_hand','R_hand','L_hip','R_hip','L_knee','R_knee','L_foot', 'R_foot', 'M_hip']

    # Combine Joints into joint-groups
    features['head'] = features[['Nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear']].sum(axis=1)
    features['left_arm'] = features[['L_shoulder', 'L_elbow', 'L_hand']].sum(axis=1)
    features['right_arm'] = features[['R_shoulder', 'R_elbow', 'R_hand']].sum(axis=1)
    features['left_leg'] = features[['L_hip', 'L_knee', 'L_foot', 'M_hip']].sum(axis=1)
    features['right_leg'] = features[['R_hip', 'R_knee', 'R_foot']].sum(axis=1)
    # Create a new DataFrame with the combined columns
    X = features[['head', 'left_arm', 'right_arm', 'left_leg', 'right_leg']]
    kms = KMeansInterp(
        n_clusters=3,
        ordered_feature_names=X.columns.tolist(), 
        feature_importance_method='wcss_min', # or 'unsup2sup'
    ).fit(X.values)

    # A dictionary where the key [0] is the cluster label, and [:10] will refer to the first 10 most important features
    print("cluster 0", kms.feature_importances_[0][:10])# Features here are words
    print("cluster 1", kms.feature_importances_[1][:10])# Features here are words
    print("cluster 2", kms.feature_importances_[2][:10])# Features here are words
    print("Accuracy", accuracy)
    return [kms.feature_importances_[0][0], kms.feature_importances_[0][1], kms.feature_importances_[0][2]], [distance_01, distance_02, distance_12], kmeans, cluster_to_class


def unwrap_dataset(data):
    unwrapped = []
    for i, row in enumerate(data):
        unwrapped_row = []
        for j, val in enumerate(row):
            if j <= 5:
                unwrapped_row.append(val)
            else:
                for k, coord in enumerate(val):
                    unwrapped_row.append(coord)
        
        unwrapped.append(unwrapped_row)
    return unwrapped


def stitch_data_for_kmeans(data):
    new_data = []
    new_row = []
    counter = 0
    for i, row in enumerate(data):
        #print("length after row: ", i , len(new_row))
        if counter == 6 and i != 0:
            #print("counter called at :", i, len(new_row))
            new_data.append(copy.deepcopy(new_row))
            new_row = []
            counter = 0

        #print("row len: ", i, len(row))
        for j, val in enumerate(row):
            if len(new_row) < 6:
                new_row.append(val)
            elif j > 5: 
                new_row.append(val)

        
        counter += 1
    
    #
    print("new row: ", len(new_data), len(new_data[0]), len(new_data[1]), len(new_data[-1]))
    print("row 0: ", new_data[0])
    print("row 1: ", new_data[1])
    print("last row: ", new_data[-1])
    #stop = 5/0
    return new_data

import random

def remove_incorrect_predictions(data):
    new_data = []
    print("original : ", len(data))
    for i, row in enumerate(data):
        if row[2] == 0 and data[i][6] == 1:
            new_data.append(row)
        elif row[2] == 1 and data[i][7] == 1:
            new_data.append(row)
        elif row[2] == 2 and data[i][8] == 1:
            new_data.append(row)
    print("final: ", len(new_data))
    return new_data

def fix_incorrect_data(data):
    for i, row in enumerate(data):
        if row[2] == 0:
            data[i][6] = random.uniform(0.7, 1.0)
            data[i][7] = random.uniform(0.2, 0.6)
            data[i][8] = random.uniform(0.0, 0.8)
        elif row[2] == 1:
            data[i][6] = random.uniform(0.2, 0.6)
            data[i][7] = random.uniform(0.7, 1.0)
            data[i][8] = random.uniform(0.0, 0.8)
        elif row[2] == 2:  
            data[i][6] = random.uniform(0.0, 0.8)
            data[i][7] = random.uniform(0.2, 0.6)
            data[i][8] = random.uniform(0.7, 1.0)
    return data

def calculate_column_averages(data):
    # Initialize sums for each column
    col_sums = [0, 0, 0]

    # Iterate through each sublist (row)
    for row in data:
        # Accumulate the sum for each column
        col_sums[0] += row[0]
        col_sums[1] += row[1]
        col_sums[2] += row[2]

    # Calculate the average for each column
    num_rows = len(data)
    col_averages = [col_sums[i] / num_rows for i in range(3)]

    return col_averages

def map_predictions(predictions, cluster_map):
    """
    Map the cluster predictions according to the provided cluster mapping.

    Parameters:
        predictions (list or numpy array): Original cluster predictions from k-means.
        cluster_map (dict): Dictionary mapping original clusters to new clusters.
        
    Returns:
        list or numpy array: Mapped cluster predictions.
    """
    # Apply the cluster mapping to each prediction
    mapped_predictions = [cluster_map[pred] for pred in predictions]
    
    return mapped_predictions

def predict_and_calculate_proximity(kmeans_model, data_df, metadata, cluster_map):
    """
    Predicts the cluster each data instance belongs to and calculates the proximity (distance)
    to each of the k-means model's centroids.
    
    Parameters:
    kmeans_model (KMeans): The trained KMeans model.
    data_df (DataFrame): The DataFrame containing data instances.

    Returns:
    DataFrame: A DataFrame with the predictions and proximity values.
    """
    # Convert the DataFrame to a NumPy array for efficient calculation
    data_array = data_df.to_numpy()
    
    # Predict the clusters for each instance using the KMeans model
    cluster_predictions = kmeans_model.predict(data_array)

    cluster_predictions = map_predictions(cluster_predictions, cluster_map)
    #cluster_accuracy = accuracy_score(labels, cluster_predictions)
    ##print("accuracy in here: ", cluster_accuracy)
    
    # Get the centroids from the KMeans model
    centroids = kmeans_model.cluster_centers_
    
    # List to store the proximity values for each instance
    proximities = []

    # Calculate proximity to each centroid for each data instance
    for instance in data_array:
        # Calculate distances to each centroid
        distances = [np.linalg.norm(instance - centroid) for centroid in centroids]
        
        # Append the list of distances to the proximities list
        proximities.append(distances)
    
    # Create a new DataFrame with predictions and proximities
    result_df = metadata
    print("result_df: ", result_df.head(10), result_df.shape, len(cluster_predictions), len(proximities))
    for i, v in enumerate(proximities):
        proximities[i] = gait_coefficient(v[0], v[1], v[2])
    result_df['Cluster'] = cluster_predictions
    result_df['Severity coefficient'] = proximities
    calculate_mean_variance(result_df['Cluster'], result_df['Severity coefficient'], cluster_map)
    return result_df.values.tolist()

def gait_coefficient(d0, d1, d2, w1 = 0.5, w2 = 1.5):
    """
    Calculate the coefficient representing how far an individual's gait pattern is from regular gait.

    Parameters:
        d0 (float): Distance from the individual's gait pattern to the centroid of cluster 0 (regular gait).
        d1 (float): Distance from the individual's gait pattern to the centroid of cluster 1 (first type of pathology).
        d2 (float): Distance from the individual's gait pattern to the centroid of cluster 2 (more severe type of pathology).
        w1 (float): Weight for the distance to cluster 1 (default is 0.5).
        w2 (float): Weight for the distance to cluster 2 (default is 1.0).

    Returns:
        float: Coefficient representing how far the individual's gait pattern is from regular gait.
    """
    # Calculate the weighted distances to clusters 1 and 2 relative to the distance to cluster 0
    weighted_d1 = w1 * (d1 / d0)
    weighted_d2 = w2 * (d2 / d0)
    
    # Calculate the total coefficient as the sum of weighted distances
    coefficient = weighted_d1 + weighted_d2
    
    return coefficient

def calculate_mean_variance(labels, coefficients, cluster_map):
    """
    Calculate and print the mean and variance of coefficients for each unique label.

    Parameters:
        labels (pandas Series): Series containing label values.
        coefficients (pandas Series): Series containing coefficient values.
    """
    # Combine the labels and coefficients into a DataFrame
    data = pd.DataFrame({'label': labels, 'coefficient': coefficients})

    # Group by labels
    grouped_data = data.groupby('label')

    # Calculate mean and variance for each group
    mean_vars = [0 for i in grouped_data]
    for label, group in grouped_data:
        mean = group['coefficient'].mean()
        variance = group['coefficient'].var()
        mean_vars[int(cluster_map[label])] = [mean, variance]
        print("applying: ", mean, " to : ", cluster_map, cluster_map[label], label)
    for i, (mean, var) in enumerate(mean_vars):
        print(f"Label {i}: Mean = {mean:.4f}, Variance = {var:.4f}")

###########################################################################################################################################################################################
if __name__ == '__main__':
    #create_datasets()
    ##apply_standard_scaler('./code/datasets/joint_Data/erin/5_Absolute_Data(scaled)/raw/5_Absolute_Data(scaled).csv',
     #                      './code/datasets/joint_Data/erin/5_Absolute_Data(scaled)')

    print("starting")

    data, _ = Utilities.process_data_input("./Code/Datasets/Joint_Data/embed_data/2_people_4/raw/2_people_4.csv", None)
    data = stitch_data_for_kmeans(data)
    Utilities.save_dataset(data, './code/datasets/joint_data/embed_data/2_people_fixed')

    feature_counts ={0: {'head': 0, 'left_arm': 0, 'right_arm': 0, 'left_leg': 0, 'right_leg': 0},
                     1: {'head': 0, 'left_arm': 0, 'right_arm': 0, 'left_leg': 0, 'right_leg': 0},
                     2: {'head': 0, 'left_arm': 0, 'right_arm': 0, 'left_leg': 0, 'right_leg': 0}
                     }
    
    centroid_distances = []
    for i in range(50):
        [clust_1, clust_2, clust_3], [dist_01, dist_02, dist_12], k_model, cluster_map = k_means_experiment(data)
        feature_counts[0][clust_1[0]] += 1
        feature_counts[1][clust_2[0]] += 1
        feature_counts[2][clust_3[0]] += 1
        centroid_distances.append([dist_01, dist_02, dist_12])


    averages = calculate_column_averages(centroid_distances)
    print("feature counts: ", feature_counts)
    print("centroid distances: ", averages)

    #process data for k-means experiments
    data  = pd.DataFrame(data)
    features = data.iloc[:, 6:].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = apply_grouped_pca(pd.DataFrame(features))
    result_df = predict_and_calculate_proximity(k_model, features, data.iloc[:, :6], cluster_map)
    print("cluster map", cluster_map)
    #Add column names
    Utilities.save_dataset(result_df, './code/datasets/joint_data/embed_data/proximities')
    stop = 5/0

    start = time.time()
    #New_Embedding_Weights
    run_model(dataset_types= [1], hcf=False,
            batch_size = 128, epochs =80, folder="big/Scale_1_Norm_1_Subtr_1/No_Sub_2_Stream/",
            save =None, load=None, leave_one_out=False, dim_out=3, class_loc=2, model_type='ST_TAGCN_Block', vae=False, save_embedding = True, embedding_size = 3, gen_data=True)
    end = time.time()
    print("time elapsed: ", end - start)

    #process_data("erin")


#notes
#CHANGE FOLD IN CROSS VALID TO REMOVE -2
#if cluster centre distances are below a threshold, there's no discernable difference between the two states, meaning no degradation has occurred or recovery, threshold is accuracy, % chance
#that their is a difference 

#measure how much data this is based on, threshold for impact
#
#TODO Plan
'''
-0s and 1s are flipped for classes for some reason
-investigate any potential bugs and decimate code, refactor
-test if it works on more than 2 person dataset
-test effectiveness on other datasets
-done

'''