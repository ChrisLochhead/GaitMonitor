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
import Programs.Machine_Learning.Clustering.clustering as clustering

#imports
import time
import random
import math
import pandas as pd
from scipy.stats import mstats
random.seed(42)
import torch
from sklearn.preprocessing import StandardScaler
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
            #weightgait dataset
            datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/13_people', 
                            '13_people.csv',             
                                joint_connections=Render.bottom_joint_connection, class_loc=class_loc, num_classes=num_classes))   
            #pathological dataset and shoedata
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/Velocity_Data', 
            #                'Velocity_Data.csv',             
            #                    joint_connections=Render.bottom_joint_connection, class_loc=class_loc, num_classes=num_classes))   
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
              save_embedding = False, embedding_size = 3, gen_data = False, dataset_name='shoe_data'):
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
                                                                     k_fold=3, batch=batch_size, epochs=epochs, device = device, gen_data = gen_data)
    else:
        model, train_scores, val_scores, test_scores, embed_data = vae_utils.cross_valid(model, multi_input_test, datasets=multi_input_train_val,
                                                                k_fold=3, batch=batch_size, epochs=epochs, device = device)
        
    if save_embedding:
        print("saving outputs ")
        embed_data = sorted(embed_data, key=lambda x: x[2])
        embed_data = reorder_instance_numbers(embed_data)
        embed_data = extract_embed_data(embed_data)
        Utilities.save_dataset(embed_data, f'./code/datasets/joint_data/embed_data/{dataset_name}_people_4')

    if save != None:
        print("saving model")
        torch.save(model.state_dict(), './Code/Datasets/Weights/' + str(save) + '.pth')
    #Process and display results
    process_results(train_scores, val_scores, test_scores)

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

###########################################################################################################################################################################################
if __name__ == '__main__':
    #create_datasets()
    #clustering.unsupervised_cluster_assessment("./Code/Datasets/Joint_Data/embed_data/Pathological_people_4/raw/Pathological_people_4.csv",
    #                                            './code/datasets/joint_data/embed_data/path_proximities', epochs=50, num_classes=6)


    embed_path = "./Code/Datasets/Joint_Data/embed_data/13_people_weightgait_people_4/raw/13_people_weightgait_people_4.csv"
    data_path = './Code/Datasets/Joint_Data/big/Scale_1_Norm_1_Subtr_1/No_Sub_2_Stream/5_people/raw/5_people.csv'
    image_path = './Code/Datasets/WeightGait/Full_Dataset/'
    means = [[] for i in range(3)]
    for i in range(1):
        epoch_means = clustering.predict_and_display(data_path, embed_path, image_path, 2, num_classes=3, normal_class=0, dataset_name='WeightGait')
        for j in range(len(means)):
            means[j].append(epoch_means[j])

    print("final mean severities: ", means)
    for i, m in enumerate(means):
        means[i] = mstats.winsorize(np.array(m), limits=[0.1, 0.1]).mean()

    print("final mean severities: ", means)
    stop = 5/0
 
    #Path paths
    '''
    embed_path = "./Code/Datasets/Joint_Data/embed_data/Pathological_people_4/raw/Pathological_people_4.csv"
    data_path = './Code/Datasets/Joint_Data/Path/Velocity_Data/raw/Velocity_Data.csv'
    image_path = './Code/Datasets/WeightGait/Full_Dataset/'
    change dim_out to 6
    change path to 'Path'
    change processing to "Velocity_Data"

    shoe data
    embed_path = "./Code/Datasets/Joint_Data/embed_data/shoe_data_people_4/raw/shoe_data_people_4.csv"
    data_path = './Code/Datasets/Joint_Data/Shoe_data/Velocity_Data/raw/Velocity_Data.csv'
    image_path = './Code/Datasets/WeightGait/Full_Dataset/'
    num classes is 9, 
    change path to "shoedata"
    change processing to velocity_data

    weightgait
    embed_path = "./Code/Datasets/Joint_Data/embed_data/2_people_4/raw/2_people_4.csv"
    data_path = './Code/Datasets/Joint_Data/big/Scale_1_Norm_1_Subtr_1/No_Sub_2_Stream/5_people/raw/5_people.csv'
    image_path = './Code/Datasets/WeightGait/Full_Dataset/'
    '''
    start = time.time()
    #New_Embedding_Weights
    run_model(dataset_types= [1], hcf=False,
            batch_size = 128, epochs =100, folder="path",
            save ='13_people', load=None, leave_one_out=False, dim_out=6, class_loc=5, model_type='ST_TAGCN_Block',
              vae=False, save_embedding = True, embedding_size = 3, gen_data=True, dataset_name='final_path')
    end = time.time()
    print("time elapsed: ", end - start)

    '''
    Folders
    weightgait: "big/scale_1_norm_1_subtr_1/no_sub_2_stream/"
    pathological
    shoe

    '''
#TODO Plan
'''
-confidence prediction is distance percentage from normal to it's prediction. Higher values correspond to more severe changes 
-accuracy is higher than just ST-TAGCN
-Importance scores tell you which joint groups are most important 

-think about how to form it into a paper
-   need to explain formula for calculating importance
    - need to see difference in confidence between overlaps in weightgait
    - need to see diff in explanation across datasets, and rationalize them with observations
-   demonstrate that the data pipeline from embedding to k-means is novel
-   not application focussed: applications in any realms of classification
-Research Question: Can traditional clustering methods be described by importance in the interest of creating more trustworthy and explainable AI
'''