from Programs.Data_Processing.Demo import *
import Programs.Data_Processing.Dataset_Creator as Creator
import Programs.Machine_Learning.GCN.Dataset_Obj as Dataset_Obj
import Programs.Data_Processing.Render as Render
import Programs.Machine_Learning.GCN.GAT as gat
import Programs.Machine_Learning.GCN.STAGCN as stgcn
import Programs.Machine_Learning.GCN.Utilities as graph_utils

import time
import random
import math
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
        Render.render_joints_series(image_data, abs_joint_data, 10, True)
        abs_joint_data = Creator.create_normalized_dataset(abs_joint_data, image_data, joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/5_Absolute_Data(norm)")
    if scale_joints:
        abs_joint_data = Creator.create_scaled_dataset(abs_joint_data, None, joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/5_Absolute_Data(scaled)")
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
                                                 output_name="./Code/Datasets/Joint_Data/" + str(folder) + "/" + str(prefix) +"_data")
    
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
    Creator.combine_datasets(rel_dum_data, vel_dum_data, None, image_data,
                                            joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/comb_data_rel_vel")
    Creator.combine_datasets(rel_dum_data, vel_dum_data, bone_dum_data, image_data,
                                            joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/comb_data_rel_vel_bone")
    Creator.combine_datasets(vel_dum_data, bone_dum_data, None, image_data,
                                             joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/comb_data_bone_vel")


def load_datasets(types, folder, multi_dim = False, num_people = 3):
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
            datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/Relative_Data', 
                                        'Relative_Data.csv',             
                                            joint_connections=Render.joint_connections_n_head))   
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/Velocity_Data', 
            #                                        'Velocity_Data.csv',             
            #                                         joint_connections=Render.joint_connections_n_head))      
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/Bones_Data', 
            #                                        'Bones_Data.csv',             
            #                                         joint_connections=Render.joint_connections_n_head))   
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
    print("Final results: ")
    for ind, t in enumerate(test_scores):
        test_scores[ind] = test_scores[ind].cpu()
        test_scores[ind] = float(test_scores[ind])

    for i, score in enumerate(train_scores):
        print("score {:.2f}: training: {:.2f}%, validation: {:.2f}%, test: {:.2f}%".format(i, score * 100, val_scores[i] * 100, test_scores[i] * 100))

    mean, var = Utilities.mean_var(test_scores)
    print("mean, std and variance: {:.2f}%, {:.2f}% {:.5f}".format(mean, math.sqrt(var), var))

def run_model(dataset_types, hcf, batch_size, epochs, folder, save = None, load = None, leave_one_out = False, multi_dim = False, num_people=3):
    #Load the full dataset
    datasets = load_datasets(dataset_types, folder, False, num_people)
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

    dim_out = 3
    if multi_dim:
        dim_out = 6
    print("\nCreating {} datasets: ".format(len(datasets)))
    print("going in: ", datasets[0].num_node_features)
    model = stgcn.GraphNetwork(dim_in=[d.num_node_features for d in datasets], dim_h=32, num_classes=dim_out, n_inputs=num_datasets,
                                data_dims=data_dims, batch_size=batch_size, hcf=hcf,
                                max_cycle=datasets[0].max_cycle, num_nodes_per_graph=datasets[0].num_nodes_per_graph, device = device, type=2)
    
    if load != None:
        print("loading model")
        model.load_state_dict(torch.load('./Code/Datasets/Weights/' + str(load) + '.pth'))

    model = model.to(device)
    model, train_scores, val_scores, test_scores = graph_utils.cross_valid(model, multi_input_test, datasets=multi_input_train_val,
                                                                     k_fold=3, batch=batch_size, epochs=epochs, device = device)
    if save != None:
        print("saving model")
        torch.save(model.state_dict(), './Code/Datasets/Weights/' + str(save) + '.pth')
    #Process and display results
    process_results(train_scores, val_scores, test_scores)

def convert_directory_to_videos(parent_folder, output, depth = False):
    image_data = []
    directory = os.fsencode(parent_folder)
    for subdir_iter, (subdir, dirs, files) in enumerate(os.walk(directory)):
        dirs.sort(key=Utilities.numericalSort)
        split = str.split(subdir.decode('utf-8'), '/')

        #avoid empty leading folders
        print("split: ", split)
        if split[-1] != "":
            print("converting to vid")
            convert_to_video(subdir.decode('utf-8'), output + split[-1], split[-1], depth=depth)


def convert_to_video(image_folder, output, file, depth = False):

    # Get the list of image files in the directory
    images = [img for img in os.listdir(image_folder) if str(img).split(".")[-1] == "jpg"]

    os.makedirs(output, exist_ok = True)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the output video file name and codec
    video_name = output + '/' + file + '.mp4'
    print("video name: ", video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 7, (width, height))
    d_video = None
    if depth:
        d_video_name = output + '/' + file + '_depth.mp4'
        print("video name: ", d_video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        d_video = cv2.VideoWriter(d_video_name, fourcc, 7, (width, height))

    for iter,  image in enumerate(images):
        if iter < len(images) / 2 or depth == False:
            image_path = os.path.join(image_folder, image)
            frame = cv2.imread(image_path)
            video.write(frame)
        else:
            #add frames to a depth video to go into the same folder destination
            image_path = os.path.join(image_folder, image)
            frame = cv2.imread(image_path)
            d_video.write(frame)

    cv2.destroyAllWindows()
    video.release()

    if d_video:
        d_video.release()

def convert_to_9_class(data, joint_output):
    data, _ = Utilities.process_data_input(data, None)
    new_data = []
    for i, row in enumerate(data):
        #check row 2, 3 and 4 and place answer in row 2
        if row[2] == 0 and row[3] == 0 and row[4] == 0:
            row[2] = 0
        elif row[2] == 0 and row[3] == 1 and row[4] == 0:
            row[2] = 1
        elif row[2] == 0 and row[3] == 0 and row[4] == 1:
            row[2] = 2
        
        elif row[2] == 1 and row[3] == 0 and row[4] == 0:
            row[2] = 3
        elif row[2] == 1 and row[3] == 1 and row[4] == 0:
            row[2] = 4
        elif row[2] == 1 and row[3] == 0 and row[4] == 1:
            row[2] = 5

        elif row[2] == 2 and row[3] == 0 and row[4] == 0:
            row[2] = 6
        elif row[2] == 2 and row[3] == 1 and row[4] == 0:
            row[2] = 7
        elif row[2] == 2 and row[3] == 0 and row[4] == 1:
            row[2] = 8

        new_data.append(row)
    
    Utilities.save_dataset(new_data, joint_output)

def create_datasets(streams = [1,2,3,4,5,6]):
        #Assign person numbers and uniform instance counts:
    folder_names = ['ahmed', 'Amy', 'Anna', 'bob', 'cade', 'emma', 'erin', 
                    'grant', 'pheobe', 'scarlett', 'sean c', 'sean g', 'wanok']
    for p in folder_names:
        #Current run: 0, 0, 0
        process_data(p, run_ims=False, norm_joints=True, scale_joints=True, subtract=True)
    
    for s in streams:
        graph_utils.stitch_dataset(folder_names=folder_names, stream=s)

def make_comb(folder, rel_path, vel_path, bone_path):
    rel, _ = Utilities.process_data_input(rel_path, None)
    vel, _ = Utilities.process_data_input(vel_path, None)
    bone, _ = Utilities.process_data_input(bone_path, None)

    Creator.combine_datasets(rel, vel, bone, None,
                                                joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/comb_data_rel_vel_bone")
    
    Creator.combine_datasets(rel, vel, None, None,
                                                joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/comb_data_rel_vel")
if __name__ == '__main__':

    #make 9d
    #convert_to_9_class('./code/datasets/joint_data/big/Scale_1_Norm_1_Subtr_1/no_Sub_4_stream/13_people/raw/13_people.csv',
     #                  "./Code/Datasets/Joint_Data/big/Scale_1_Norm_1_Subtr_1/multi_dim_3_13" )
    #stop = 5/0

    #Dataset creator
    process_data("Anna")
    #create_datasets()
    folder_names = ['ahmed', 'Amy', 'Anna', 'bob', 'cade', 'emma', 'erin', 
                    'grant', 'pheobe', 'scarlett', 'sean c', 'sean g', 'wanok']
    
    #for f in folder_names:
    #    make_comb(f, './code/datasets/joint_data/' + str(f) + "/rel_data/raw/rel_data.csv",
    #              './code/datasets/joint_data/' + str(f) + "/vel_data/raw/vel_data.csv",
    #                './code/datasets/joint_data/' + str(f) + "/bone_data/raw/bone_data.csv"  )

    #graph_utils.stitch_dataset(folder_names, stream = 5)
    #graph_utils.stitch_dataset(folder_names, stream = 6)
    #stop = 5/0
    #Run the model:
    #Dataset types: Array of types for the datasets you want to pass through at the same time
    #   1: normal full body 9D dataset
    #   2: HCF data
    #   3: 2 region data
    #   4: 5 region data
    #Cycles: is the data passed through as single frames or gait cycles
    #Model type: "ST-AGCN" or "GAT"
    #hcf: indicates presence of an HCF dataset
    #multi: indicates if the multi-stream variant of the chosen model should be used (multi variant 
    # models are compatible with both single and multiple datasets)
    #Leave_one_out: indicates whether using normally split data or data split by person
    #Person: full dataset only, denotes which person to extract otherwise 0 or none.
    #Label: which label to classify by: 2 = gait type, 3 = freeze, 4 = obstacle, 5 = person (not implemented)

    #WHAT AM I DOING:Cust-stgcn, then gaitgraph2 (switch to 3 streams)

    #paper: add new numbers to tables, shorten to 14 pages, done :)
    start = time.time()
    run_model(dataset_types= [1], hcf=False,
            batch_size = 128, epochs = 80, folder="path",
            save =None, load=None, leave_one_out=False, multi_dim=True, num_people=13)
    end = time.time()
    print("time elapsed: ", end - start)

    #Save all outputs to files for f1 calcs etc
    
    #So my best version: bone + vel needs to compete with: (all early-fusion) (w best of scaling, norm and subtraction) (1 day)
        #2-stream but with st-tagcn (rel + vel)
        #3-stream but with st-tagcn (rel + vel + bone )
    
    #Then my best version of these needs to be tested with (bone + vel, scaling norm and subtr) (1 day)
        #Resnet
        #SVM
        #KNN
        #st-gcn (just a different model)
        #st-jagcn (early fusion rel + vel + bone)
        #2-st-gcn (2 separate streams rel + vel using st-gcn)
        #Mine
    
    #Then my results on best version have to compete with (1 week)
        #Azure datasets
            #normal
            #leave-one-out
        #Foot platform dataset
            #normal
            # leave one out
    
    #DONE
    #Then I need one ablation to make sure it generalizes to different ailments for each person
        #Results are: 50% on 3 person, 55% on 4 people, 57% on 5 people 61.2% on 13 people (leaving one out) 
        