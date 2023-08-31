#import init_directories
from Programs.Data_Processing.Model_Based.Demo import *
#from Programs.Data_Processing.Model_Based.Utilities import load, load_images, save_dataset
import Programs.Data_Processing.Model_Based.Dataset_Creator as Creator
import Programs.Machine_Learning.Model_Based.AutoEncoder.GAE as GAE
import Programs.Machine_Learning.Model_Based.GCN.Dataset_Obj as Dataset_Obj
import Programs.Machine_Learning.Model_Based.GCN.Ground_Truths as GT
import Programs.Data_Processing.Model_Based.Render as Render
import torch
torch.manual_seed(42)

#import torch_geometric
import random
random.seed(42)
import Programs.Machine_Learning.Model_Based.GCN.GAT as gat
import Programs.Machine_Learning.Model_Based.GCN.STAGCN as stgcn
import Programs.Machine_Learning.Model_Based.GCN.Utilities as graph_utils
torch.cuda.empty_cache()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
def process_data(folder = "Chris"):

############################################# PIPELINE ##################################################################

    #Extract joints from images
    #run_images("./Code/Datasets/" + str(folder) + "/Full_Dataset", out_folder="./Code/Datasets/Joint_Data/" + str(folder) + "/", exclude_2D=False, 
    #          start_point=-1)
    
    #Display first 2 instances of results 
    #print("\nStage 1: ")
    #render_joints_series("./Code/Datasets/WeightGait/Raw_Images", joints=abs_joint_data,
    #                     size = 20, delay=True, use_depth=True)

    #Remove empty frames
    #abs_joint_data, image_data = Creator.process_empty_frames(joint_file="./Code/Datasets/Joint_Data/" + str(folder) + "/Absolute_Data.csv",
    #                                             image_file="./Code/Datasets/" + str(folder) + "/Full_Dataset/",
    #                                             joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/2_Absolute_Data(empty frames removed)",
    #                                             image_output="./Code/Datasets/" + str(folder) + "/2_Empty Frames Removed/")

    #Display first 2 instances of results
    print("\nStage 2: ")
    #render_joints_series(image_data, abs_joint_data, size=15)
    #render_joints_series(image_data, abs_joint_data, size=15, plot_3D=True)
    
    #Trim start and end frames where joints get confused by image borders
    #abs_joint_data, image_data =Creator.process_trimmed_frames(abs_joint_data, image_data,
    #                                                    joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/3_Absolute_Data(trimmed instances)",
    #                                                    image_output="./Code/Datasets/" + str(folder) + "/3_Trimmed Instances/", trim = 5)

    abs_joint_data, image_data = Utilities.process_data_input("./Code/Datasets/Joint_Data/" + str(folder) + "/3_Absolute_Data(trimmed instances)/raw/3_Absolute_Data(trimmed instances).csv",
                                                              "./Code/Datasets/" + str(folder) + "/3_Trimmed Instances/", cols=Utilities.colnames, ignore_depth=False)
    if folder == 'weightgait':
        abs_joint_data = Utilities.fix_multi_person_labels(abs_joint_data, 
                                                        joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/4_Absolute_Data(class_fixed)")
    #done = 5/0
    print("\nStage 4:")
    #render_joints_series(image_data, abs_joint_data, size=10)
    abs_joint_data = Creator.append_midhip(abs_joint_data, image_data, 
                                                   joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/5_Absolute_Data(midhip)")

    #Create relative dataset
    relative_joint_data = Creator.create_relative_dataset(abs_joint_data, image_data,
                                                 joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/6_Relative_Data(relative)")

    #Flip all the joints to be facing one way to prepare for background skeleton subtraction
    relative_joint_data = Creator.create_flipped_joint_dataset(relative_joint_data, abs_joint_data, None,
                                                                joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/12_Flipped(same_way)", double_size=False)
    relative_joint_data = Creator.subtract_skeleton(relative_joint_data, 
                                                    joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/12_Rel_Subtracted",
                                                      base_output="./Code/Datasets/Joint_Data/" + str(folder) + "/12_Rel_base")
    
 
    relative_joint_data = Creator.create_flipped_joint_dataset(Utilities.convert_to_sequences(relative_joint_data),
                                                                Creator.interpolate_gait_cycle(Utilities.convert_to_sequences(abs_joint_data),
                                                                None, 0, restrict_cycle=True), None,
                                                                joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Rel_Flipped(double)",
                                                                double_size=True, already_sequences=True)
    rel_dummy = Creator.create_dummy_dataset(relative_joint_data, 
                                                output_name="./Code/Datasets/Joint_Data/" + str(folder) + "/20_Rel_Data_Noise")

    #Create velocity dataset
    velocity_data = Creator.create_velocity_dataset(abs_joint_data, image_data, 
                                                    joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Velocity_Data(velocity)")
    #Flip all the joints to be facing one way to prepare for background skeleton subtraction
    velocity_data = Creator.create_flipped_joint_dataset(velocity_data, abs_joint_data, None,
                                                                joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Vel_Flipped(same_way)", double_size=False)
    velocity_data = Creator.subtract_skeleton(velocity_data, 
                                                    joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Vel_Subtracted",
                                                      base_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Vel_base")
    velocity_data = Creator.create_flipped_joint_dataset(Utilities.convert_to_sequences(velocity_data),
                                                                Creator.interpolate_gait_cycle(Utilities.convert_to_sequences(abs_joint_data),
                                                                None, 0, restrict_cycle=True), None,
                                                                joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Vel_Flipped(double)",
                                                                double_size=True, already_sequences=True)
    vel_dummy = Creator.create_dummy_dataset(velocity_data, 
                                            output_name="./Code/Datasets/Joint_Data/" + str(folder) + "/20_Vel_Data_Noise")
    #Create joint angles data
    joint_bones_data = Creator.create_bone_dataset(abs_joint_data, image_data, 
                                                   joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/15_Bones_Data(integrated)")
    #Flip all the joints to be facing one way to prepare for background skeleton subtraction
    joint_bones_data = Creator.create_flipped_joint_dataset(joint_bones_data, abs_joint_data, None,
                                                                joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/14_Bone_Flipped(same_way)", double_size=False)
    joint_bones_data = Creator.subtract_skeleton(joint_bones_data, 
                                                    joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/14_Bone_Subtracted",
                                                      base_output="./Code/Datasets/Joint_Data/" + str(folder) + "/14_Bone_base")
    joint_bones_data = Creator.create_flipped_joint_dataset(Utilities.convert_to_sequences(joint_bones_data),
                                                                Creator.interpolate_gait_cycle(Utilities.convert_to_sequences(abs_joint_data),
                                                                None, 0, restrict_cycle=True), None,
                                                                joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Bone_Flipped(double)",
                                                                double_size=True, already_sequences=True)
    joint_bone_dummy = Creator.create_dummy_dataset(joint_bones_data, 
                                                output_name="./Code/Datasets/Joint_Data/" + str(folder) + "/20_Bone_Data_Noise")
    #render_velocity_series(abs_joint_data, joint_bones_data, image_data, size=20)

    combined_data = Creator.combine_datasets(relative_joint_data, velocity_data, joint_bones_data, image_data,
                                             joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/19_Combined_Data")

    combined_data = Creator.create_dummy_dataset(combined_data, 
                                                 output_name="./Code/Datasets/Joint_Data/" + str(folder) + "/20_Combined_Data_Noise")

    
    if folder == "weightgait":
        bob = Creator.create_n_size_dataset(combined_data, joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/Bob", n=[3])
        chris_elisa = Creator.create_n_size_dataset(combined_data, joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/CE", n=[5,6])
        chris_elisa = Creator.create_n_size_dataset(combined_data, joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/CEB", n=[3,5,6])

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
                                            '16_Combined_Data_2Region_top.csv',
                                            joint_connections=joints[i]))
    return datasets

#Types are 1 = normal, 2 = HCF, 3 = 2 region, 4 = 5 region, 5 = Dummy. Pass types as an array of type numbers, always put hcf (2) at the END if including. If including HCF, you MUST include
#as cycles = True
def load_datasets(types, folder):
    datasets = []
    print("loading datasets...")
        
    for i, t in enumerate(types):
        print("loading dataset {} of {}. ".format(i + 1, len(types)), t)
        #Type 1: Normal, full dataset
        if t == 1:  
            #15.5 COMBINED DATASET
            datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/20_Combined_Data_Noise', '20_Combined_Data_Noise.csv',
                                                  joint_connections=Render.joint_connections_n_head))
            
            #Experimental
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/20_Rel_Data_Noise', '20_Rel_Data_Noise.csv',
            #                                       joint_connections=Render.joint_connections_m_hip, cycles=True))
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/20_Vel_Data_Noise', '20_Vel_Data_Noise.csv',
            #                                      joint_connections=Render.joint_connections_m_hip, cycles=True))
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/20_Bone_Data_Noise', '20_Bone_Data_Noise.csv',
            #                                       joint_connections=Render.joint_connections_m_hip, cycles=True))
        #Type 2: HCF dataset
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
    #Return requested datasets
    return datasets

def get_balanced_samples(dataset, train = 0.9, test = 0.1):
    print(len(dataset))
    class_size = int(len(dataset) / 3)
    print("class size: ", class_size)

    train_indices = []
    test_indices = []

    train_class_size = int(class_size * 0.9)
    to_fill = [train_class_size, train_class_size, train_class_size]
    print("fill sizes: ", to_fill)
    for i, cycle in enumerate(dataset):
        #print("cycle: ", cycle.y.item())
        if to_fill[cycle.y.item()] > 0:
            to_fill[cycle.y.item()] -= 1
            train_indices.append(i)
    
    for i in range(len(dataset)):
        if i not in train_indices:
            test_indices.append(i)

    print("to fill should be empty: ", to_fill)
    print("train indices: ", len(dataset), len(train_indices))
    print("len test: ", len(test_indices))

    print("indices: ", train_indices)
    print("test: ", test_indices)

    print("done")
    return train_indices, test_indices


def process_datasets(datasets):
    print("Processing data...")

    train_indice_list = []
    test_indice_list = []

    for dataset in datasets:
        dataset_size = len(dataset)

        #Append indices based on the first dataset length
        train_indices = random.sample(range(dataset_size), int(0.9 * dataset_size))
        print("original indices:", len(train_indices) )
        test_indices = random.sample(set(range(dataset_size)) - set(train_indices), int(0.1 * dataset_size))
        print("original test indices:", len(test_indices) )
        #train_indices, test_indices = get_balanced_samples(datasets[0])
        train_indice_list.append(train_indices)
        test_indice_list.append(test_indices)

    #These regions will be the same for both datasets
    multi_input_train_val = []
    multi_input_test = []
    for i, dataset in enumerate(datasets):
        multi_input_train_val.append(dataset[train_indice_list[i]])
        multi_input_test.append(dataset[test_indice_list[i]])
    
    print("indices lens so number of examples per: train: ", len(multi_input_train_val[0]), len(multi_input_test[0]))
    print("Dataset processing complete.")

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

def run_model(dataset_types, model_type, hcf, batch_size, epochs, folder):

    #Load the full dataset alongside HCF with gait cycles
    datasets = load_datasets(dataset_types, folder)
    print("datasets here: ", datasets)
    #Concatenate data dimensions for ST-GCN
    data_dims = []
    for dataset in datasets:
        data_pair = [dataset.num_features, dataset.num_node_features]
        data_dims.append(data_pair)

    num_datasets = len(datasets)

    print("dataset info: ", len(datasets[0]), dataset[0])
    #done = 5/0

    #Accounting for extra original dataset not used in dummy case for training but only for testing
    if 5 in dataset_types:
        num_datasets -= 1

    print("number of datasets: ", num_datasets)
    
    #Split classes by just making the last person the test set and the rest training and validation.
    #if leave_one_out:
    #    multi_input_train_val, multi_input_test = graph_utils.split_data_by_person(datasets)
    #else:
        #Process datasets by manually shuffling to account for cycles
    multi_input_train_val, multi_input_test = process_datasets(datasets)

    dim_out = 3

    print("\nCreating {} model with {} datasets: ".format(model_type, len(datasets)))
    if model_type == "GAT":
        model = gat.MultiInputGAT(dim_in=[d.num_node_features for d in datasets], dim_h=64, dim_out=dim_out, hcf=hcf, n_inputs=num_datasets)
    elif model_type == "ST-AGCN":
        print("going in: ", datasets[0].num_node_features)
        model = stgcn.MultiInputSTGACN(dim_in=[d.num_node_features for d in datasets], dim_h=32, num_classes=dim_out, n_inputs=num_datasets,
                                    data_dims=data_dims, batch_size=batch_size, hcf=hcf,
                                    max_cycle=datasets[0].max_cycle, num_nodes_per_graph=datasets[0].num_nodes_per_graph, device = device)
    else:
        print("Invalid model type.")
        return
    model = model.to(device)

    train_scores, val_scores, test_scores = graph_utils.cross_valid(model, multi_input_test, datasets=multi_input_train_val,
                                                                     k_fold=2, batch=batch_size, epochs=epochs, type=model_type)

    #Process and display results
    process_results(train_scores, val_scores, test_scores)

if __name__ == '__main__':
    #
    #process_data("erin")
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

    run_model(dataset_types= [1], model_type = "ST-AGCN", hcf=False,
           batch_size = 64, epochs = 100, folder="big")