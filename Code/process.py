#import init_directories
from Programs.Data_Processing.Model_Based.Demo import *
import Programs.Data_Processing.Model_Based.Dataset_Creator as Creator
import Programs.Machine_Learning.Model_Based.GCN.Dataset_Obj as Dataset_Obj
import Programs.Data_Processing.Model_Based.Render as Render
import Programs.Machine_Learning.Model_Based.GCN.GAT as gat
import Programs.Machine_Learning.Model_Based.GCN.STAGCN as stgcn
import Programs.Machine_Learning.Model_Based.GCN.Utilities as graph_utils

import time
import torch
torch.manual_seed(42)
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import random
random.seed(42)

def process_data(folder = "Chris"):
############################################# PIPELINE ##################################################################

    #Extract joints from images
    print("\nStage 1: Extracting images ")
    #run_images("./Code/Datasets/Individuals/" + str(folder) + "/Full_Dataset", out_folder="./Code/Datasets/Joint_Data/" + str(folder) + "/", exclude_2D=False, 
    #          start_point=0)
    
    #Display first 2 instances of results 
    #render_joints_series("./Code/Datasets/WeightGait/Raw_Images", joints=abs_joint_data,
    #                     size = 20, delay=True, use_depth=True)

    #Remove empty frames
    print("\nStage 2: Removing empty frames.")
    #abs_joint_data, image_data = Creator.process_empty_frames(joint_file="./Code/Datasets/Joint_Data/" + str(folder) + "/Absolute_Data.csv",
    #                                             image_file="./Code/Datasets/Individuals/" + str(folder) + "/Full_Dataset/",
    #                                             joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/2_Absolute_Data(empty frames removed)",
    #                                             image_output="./Code/Datasets/Individuals/" + str(folder) + "/2_Empty Frames Removed/")

    #Display first 2 instances of results
    print("\nStage 3:  Trimming sequences")
    #render_joints_series(image_data, abs_joint_data, size=15)
    #render_joints_series(image_data, abs_joint_data, size=15, plot_3D=True)
    #abs_joint_data, image_data = Utilities.process_data_input("./Code/Datasets/Joint_Data/" + str(folder) + "/2_Absolute_Data(empty frames removed)/raw/2_Absolute_Data(empty frames removed).csv",
    #                                                          "./Code/Datasets/" + str(folder) + "/2_Empty Frames Removed/", cols=Utilities.colnames, ignore_depth=False)
    
    #Trim start and end frames where joints get confused by image borders
    #abs_joint_data, image_data =Creator.process_trimmed_frames(abs_joint_data, image_data,
    #                                                    joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/3_Absolute_Data(trimmed instances)",
    #                                                     image_output="./Code/Datasets/Individuals/" + str(folder) + "/3_Trimmed Instances/", trim = 5)

    print("\nStage 4: Reloading data into memory (shortcut)")
    abs_joint_data, image_data = Utilities.process_data_input("./Code/Datasets/Joint_Data/" + str(folder) + "/3_Absolute_Data(trimmed instances)/raw/3_Absolute_Data(trimmed instances).csv",
                                                              "./Code/Datasets/Individuals/" + str(folder) + "/3_Trimmed Instances/", cols=Utilities.colnames, ignore_depth=False)

    
    #Utilities.save_images(abs_joint_data, copy.deepcopy(image_data), directory="./Code/Datasets/PaperImages/Imperfect" + str(folder) + "/", 
    #                      include_joints=True, aux_joints=abs_joint_data)

    print("\nStage 5: Adding midhip data")
    abs_joint_data = Creator.append_midhip(abs_joint_data, image_data, 
                                                   joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/5_Absolute_Data(midhip)")
    #render_joints_series(image_data, abs_joint_data, size=25)
    imperfect_joints = copy.deepcopy(abs_joint_data)
    
    #Removes various types of outliers
    print("\nStage 5: Normalizing outliers")
    abs_joint_data = Creator.create_normalized_dataset(abs_joint_data, image_data, joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/5_Absolute_Data(norm)")
    #render_joints_series(image_data, abs_joint_data, size=25)

    print("saving images: ")
    #Utilities.save_images(abs_joint_data, copy.deepcopy(image_data), directory="./Code/Datasets/PaperImages/" + str(folder) + "/", include_joints=True, aux_joints = imperfect_joints)
    #done = 5/0

    print("\nStage 6: Standardizing data scales")
    abs_joint_data = Creator.create_scaled_dataset(abs_joint_data, None, joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/5_Absolute_Data(scaled)")
    #render_joints_series(image_data, abs_joint_data, size=10)
    #Utilities.save_images(abs_joint_data, copy.deepcopy(image_data), directory="./Code/Datasets/PaperImages/Scaled/" + str(folder) + "/", include_joints=True, aux_joints = None)


    #Create relative dataset
    print("\nStage 7: Relativizing data")
    relative_joint_data = Creator.create_relative_dataset(abs_joint_data, image_data,
                                                 joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/6_Relative_Data(relative)")

    rel_dum_data = Creator.create_dummy_dataset(relative_joint_data, 
                                                 output_name="./Code/Datasets/Joint_Data/" + str(folder) + "/6_5_Rel_Data_Noise")
    
    #Flip all the joints to be facing one way to prepare for background skeleton subtraction
    print("\nStage 8: Subtraction and flipping: relative data.")
    relative_joint_data = Creator.create_flipped_joint_dataset(relative_joint_data, abs_joint_data, None,
                                                                joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/12_Flipped(same_way)", double_size=False)
    relative_joint_data = Creator.subtract_skeleton(relative_joint_data, 
                                                    joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/12_Rel_Subtracted",
                                                      base_output="./Code/Datasets/Joint_Data/" + str(folder) + "/12_Rel_base")
    
 
    relative_joint_data = Creator.create_flipped_joint_dataset(Utilities.convert_to_sequences(relative_joint_data),
                                                                Creator.interpolate_gait_cycle(Utilities.convert_to_sequences(abs_joint_data),
                                                                None, 0, restrict_cycle=False), None,
                                                                joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Rel_Flipped(double)",
                                                                double_size=True, already_sequences=True)

    #Create velocity dataset
    print("\nStage 9: Subtraction and flipping: velocity data.")
    velocity_data = Creator.create_velocity_dataset(abs_joint_data, image_data, 
                                                    joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Velocity_Data(velocity)")
    
    vel_dum_data = Creator.create_dummy_dataset(relative_joint_data, 
                                                output_name="./Code/Datasets/Joint_Data/" + str(folder) + "/13_5_Vel_Data_Noise")
    
    #Flip all the joints to be facing one way to prepare for background skeleton subtraction
    velocity_data = Creator.create_flipped_joint_dataset(velocity_data, abs_joint_data, None,
                                                                joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Vel_Flipped(same_way)", double_size=False)
    velocity_data = Creator.subtract_skeleton(velocity_data, 
                                                    joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Vel_Subtracted",
                                                      base_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Vel_base")
    velocity_data = Creator.create_flipped_joint_dataset(Utilities.convert_to_sequences(velocity_data),
                                                                Creator.interpolate_gait_cycle(Utilities.convert_to_sequences(abs_joint_data),
                                                                None, 0, restrict_cycle=False), None,
                                                                joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Vel_Flipped(double)",
                                                                double_size=True, already_sequences=True)

    #Create joint angles data
    print("\nStage 10: Subtraction and flipping: bones data.")
    joint_bones_data = Creator.create_bone_dataset(abs_joint_data, image_data, 
                                                   joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/15_Bones_Data(integrated)")
    

    bone_dum_data = Creator.create_dummy_dataset(joint_bones_data, 
                                                 output_name="./Code/Datasets/Joint_Data/" + str(folder) + "/15_5_Rel_Data_Noise")
    
    #Flip all the joints to be facing one way to prepare for background skeleton subtraction
    joint_bones_data = Creator.create_flipped_joint_dataset(joint_bones_data, abs_joint_data, None,
                                                                joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/14_Bone_Flipped(same_way)", double_size=False)
    joint_bones_data = Creator.subtract_skeleton(joint_bones_data, 
                                                    joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/14_Bone_Subtracted",
                                                      base_output="./Code/Datasets/Joint_Data/" + str(folder) + "/14_Bone_base")
    joint_bones_data = Creator.create_flipped_joint_dataset(Utilities.convert_to_sequences(joint_bones_data),
                                                                Creator.interpolate_gait_cycle(Utilities.convert_to_sequences(abs_joint_data),
                                                                None, 0, restrict_cycle=False), None,
                                                                joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Bone_Flipped(double)",
                                                                double_size=True, already_sequences=True)

    print("\nStage 11: Create 9D-1-Stream dataset")
    #combined_data = Creator.combine_datasets(relative_joint_data, velocity_data, joint_bones_data, image_data,
    #                                         joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/19_Combined_Data")

    com_dum_data = Creator.combine_datasets(rel_dum_data, bone_dum_data, None, image_data,
                                             joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/19_5_Combined_Data")
    
    #combined_data = Creator.create_dummy_dataset(combined_data, 
    #                                             output_name="./Code/Datasets/Joint_Data/" + str(folder) + "/20_Combined_Data_Noise")

#Currently unused: Load data in by body part
def load_region_data(folder, type):
    if type == 3:
        paths = ['2_Region_top', '2_Region_bottom']
        joints = [Render.top_joint_connections, Render.bottom_joint_connection]
    elif type == 4:
        paths = ['l_leg', 'r_leg', 'l_arm', 'r_arm', 'head']
        joints = [Render.limb_connections, Render.limb_connections, Render.limb_connections, Render.limb_connections, Render.head_joint_connections]

    datasets = []
    for i, path in enumerate(paths):
        datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/' + path, path + ".csv",
                                            joint_connections=joints[i]))
    return datasets

#Types are 1 = normal, 2 = HCF, 3 = 2 region, 4 = 5 region, 5 = Dummy. Pass types as an array of type numbers, always put hcf (2) at the END if including. If including HCF, you MUST include
#as cycles = True
def load_datasets(types, folder, multi_dim = False):
    datasets = []
    print("loading datasets...")
        
    for i, t in enumerate(types):
        print("loading dataset {} of {}. ".format(i + 1, len(types)), t)
        #Type 1: Normal, full dataset
        if t == 1:  
            #9D-1-Stream
            datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/5_people',
                                                    '5_people.csv',             
                                                  joint_connections=Render.joint_connections_n_head))          
            
            #3D-3-Stream
            #Experimental
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/no_subtracted/15_rel', '15_rel.csv',
            #                                       joint_connections=Render.joint_connections_n_head))
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/no_subtracted/15_vel', '15_vel.csv',
            #                                      joint_connections=Render.joint_connections_n_head))
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/15_bone', '15_bone.csv',
            #                                       joint_connections=Render.joint_connections_n_head))
        #Type 2: HCF dataset (unused)
        elif t == 2:
            #This MUST have cycles, there's no non-cycles option
            dataset = Dataset_Obj.HCFDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/13.5_HCF_Data(normed)',
                                                    '13.5_HCF_Data(normed).csv', cycles=True)
            datasets.append(dataset)
        #Type 3: 2 region, Type 4: 5 region
        elif t == 3 or t == 4:
            region_datasets = load_region_data(folder, t)
            for data in region_datasets:
                datasets.append(data)

    print("datasets loaded.")
    if multi_dim:
        for index, dataset in enumerate(datasets):
            datasets[index] = Creator.convert_person_to_type(dataset, None)

    #Return requested datasets
    return datasets

#Balance the number of examples per class
def get_balanced_samples(dataset, train = 0.9, test = 0.1):
    print(len(dataset))
    class_size = int(len(dataset) / 3)
    train_indices = []
    test_indices = []

    train_class_size = int(class_size * 0.9)
    to_fill = [train_class_size, train_class_size, train_class_size]
    for i, cycle in enumerate(dataset):
        if to_fill[cycle.y.item()] > 0:
            to_fill[cycle.y.item()] -= 1
            train_indices.append(i)
    
    for i in range(len(dataset)):
        if i not in train_indices:
            test_indices.append(i)

    return train_indices, test_indices

def leave_one_out_dataset(datasets):
    #Just make train and test from their own datasets, this will always be 2, the first dataset will be train, the second test.
    train_indices = random.sample(range(len(datasets[0])), int(len(datasets[0])))
    test_indices = random.sample(set(range(len(datasets[1]))), int(len(datasets[1])))
    #These regions will be the same for both datasets
    #Keep as an array of arrays so it's structurally consistent with the rest of the code
    multi_input_train_val = []
    multi_input_test = []
    multi_input_train_val.append(datasets[0][train_indices])
    multi_input_test.append(datasets[1][test_indices])

    return multi_input_train_val, multi_input_test

def process_datasets(datasets):
    print("Processing data...")
    train_indice_list = []
    test_indice_list = []
    for dataset in datasets:
        dataset_size = len(dataset)

        #Append indices based on the first dataset length
        train_indices = random.sample(range(dataset_size), int(0.9 * dataset_size))
        test_indices = random.sample(set(range(dataset_size)) - set(train_indices), int(0.1 * dataset_size))
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

def run_model(dataset_types, hcf, batch_size, epochs, folder, save = None, load = None, leave_one_out = False, multi_dim = False):
    #Load the full dataset
    datasets = load_datasets(dataset_types, folder, multi_dim)
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
    if leave_one_out == False:
        multi_input_train_val, multi_input_test = process_datasets(datasets)
    else:
        multi_input_train_val, multi_input_test = leave_one_out_dataset(datasets)
        datasets = [datasets[0]]
        num_datasets = len(datasets)

    dim_out = 3
    if multi_dim:
        dim_out = 9
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
                                                                     k_fold=5, batch=batch_size, epochs=epochs, device = device)
    if save != None:
        print("saving model")
        torch.save(model.state_dict(), './Code/Datasets/Weights/' + str(save) + '.pth')
    #Process and display results
    process_results(train_scores, val_scores, test_scores)

def convert_to_video(image_folder, output):

    # Get the list of image files in the directory
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    # Sort the images in the desired order (if needed)
    #images.sort()

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the output video file name and codec
    video_name = output + '.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 4, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':

    #process_data('bob')
    #process_data('cade')
    #process_data('emma')
    #process_data('scarlett')
    #process_data('sean c')
    #process_data('sean g')
    #Assign person numbers and uniform instance counts:
    folder_names = ['bob', 'cade', 'emma', 'scarlett', 'sean c', 'sean g']
     
    #graph_utils.stitch_dataset(folder_names=folder_names)
    #data, _ = Utilities.process_data_input('./Code/Datasets/Joint_Data/Big/no_subtracted/15_people/raw/15_people.csv', None, 
    #                                       cols=Utilities.colnames_nohead, ignore_depth=False)

    #Split 9D into 3 stream representation
    #Creator.split_into_streams(data, joint_output_r='./Code/Datasets/Joint_Data/Big/no_subtracted/15_rel',
    #                           joint_output_v='./Code/Datasets/Joint_Data/Big/no_subtracted/15_vel',
    #                           joint_output_b='./Code/Datasets/Joint_Data/Big/no_subtracted/15_bone')

    
    #convert_to_video('./Code/Datasets/PaperImages/ImperfectChris/Instance_0.0', './Code/Datasets/PaperImages/Videos/Chris_imperf_0')
    #convert_to_video('./Code/Datasets/PaperImages/Scaled/Chris/Instance_0.0', './Code/Datasets/PaperImages/Videos/Chris_0')

    #process_data('bob')

    #abs_joint_data, _ = Utilities.process_data_input("./Code/Datasets/Joint_Data/Chris/6_Relative_Data(relative)/raw/6_Relative_Data(relative).csv",
    #                                                          None, cols=Utilities.colnames_midhip, ignore_depth=False)
    #Creator.compute_joint_stats(abs_joint_data)

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
    #start = time.time()
    run_model(dataset_types= [1], hcf=False,
           batch_size = 64, epochs = 150, folder="big", save =None, load=None, leave_one_out = False, multi_dim=False)#

    #end = time.time()
    #print("time elapsed: ", end - start)


    #2s Instant fusing might work per that other 2s work, showing 3s is shit novel part of network p1 and novel representation
    # Novel network joint attention, but needs one other thing I think
    #3 is weightgait which is strong


    #TODO to turn this into a regression problem paper
    #replace output with 2 values softmaxxed, probabilities of belonging to 0 or 1
    #if it doesn't work immidiately, experiment with autoencoder to embed all models into lower latent space and reduce required data and resources
        #Maybe ST-GCN autoencoder to encode temporal elements, add attention too?
        #Use autoencoder to fling in HCF data like time of day and speed

    #implement way to decide which joints have the most impact on classification
    #record new metadata such as time of day
    #implement how to show how far into gait cycle an issue is detected and draw the result
    #aim to predict time of day, whether fall is legitimate or caused by environment
    #chart speed, gait length 
    #Build average of two discerned classes and chart differences in speed, gait length, frequencies of issues, TOD walking etc as a visible chart
    #Use as a regressor between closer to which assessment period

    #Once this is done, write final technical paper on novel ST-GCN with autoencoder using HCF
    #DONE :D

    #Ask Bob: How Do I ensure there is a difference between the 2? artificially change them i.e. how many times walking at night in 1 vs other, simulate recovering from an injury
    #in the second one?



    