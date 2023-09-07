#import init_directories
from Programs.Data_Processing.Model_Based.Demo import *
#from Programs.Data_Processing.Model_Based.Utilities import load, load_images, save_dataset
import Programs.Data_Processing.Model_Based.Dataset_Creator as Creator
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
    #run_images("./Code/Datasets/Individuals/" + str(folder) + "/Full_Dataset", out_folder="./Code/Datasets/Joint_Data/" + str(folder) + "/", exclude_2D=False, 
    #          start_point=0)
    
    #Display first 2 instances of results 
    print("\nStage 1: ")
    #render_joints_series("./Code/Datasets/WeightGait/Raw_Images", joints=abs_joint_data,
    #                     size = 20, delay=True, use_depth=True)

    #Remove empty frames
    #abs_joint_data, image_data = Creator.process_empty_frames(joint_file="./Code/Datasets/Joint_Data/" + str(folder) + "/Absolute_Data.csv",
    #                                             image_file="./Code/Datasets/Individuals/" + str(folder) + "/Full_Dataset/",
    #                                             joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/2_Absolute_Data(empty frames removed)",
    #                                             image_output="./Code/Datasets/Individuals/" + str(folder) + "/2_Empty Frames Removed/")

    #Display first 2 instances of results
    print("\nStage 2: ")
    #render_joints_series(image_data, abs_joint_data, size=15)
    #render_joints_series(image_data, abs_joint_data, size=15, plot_3D=True)
    
    #Trim start and end frames where joints get confused by image borders
    #abs_joint_data, image_data =Creator.process_trimmed_frames(abs_joint_data, image_data,
    #                                                    joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/3_Absolute_Data(trimmed instances)",
    #                                                     image_output="./Code/Datasets/Individuals/" + str(folder) + "/3_Trimmed Instances/", trim = 5)

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
    
    #render_joints_series(image_data, abs_joint_data, size=25)

    #This normalization just removed outliers
    abs_joint_data = Creator.create_normalized_dataset(abs_joint_data, None, joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/5_Absolute_Data(norm)")

    #render_joints_series(image_data, abs_joint_data, size=25)
    #done = 5/0

    abs_joint_data = Creator.create_scaled_dataset(abs_joint_data, None, joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/5_Absolute_Data(scaled)")

    #render_joints_series(image_data, abs_joint_data, size=10)


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
                                                                None, 0, restrict_cycle=False), None,
                                                                joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Rel_Flipped(double)",
                                                                double_size=True, already_sequences=True)
    #rel_dummy = Creator.create_dummy_dataset(relative_joint_data, 
    #                                            output_name="./Code/Datasets/Joint_Data/" + str(folder) + "/20_Rel_Data_Noise")

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
                                                                None, 0, restrict_cycle=False), None,
                                                                joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Vel_Flipped(double)",
                                                                double_size=True, already_sequences=True)
    #vel_dummy = Creator.create_dummy_dataset(velocity_data, 
    #                                        output_name="./Code/Datasets/Joint_Data/" + str(folder) + "/20_Vel_Data_Noise")
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
                                                                None, 0, restrict_cycle=False), None,
                                                                joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_Bone_Flipped(double)",
                                                                double_size=True, already_sequences=True)
    #joint_bone_dummy = Creator.create_dummy_dataset(joint_bones_data, 
    #                                            output_name="./Code/Datasets/Joint_Data/" + str(folder) + "/20_Bone_Data_Noise")
    #render_velocity_series(abs_joint_data, joint_bones_data, image_data, size=20)

    combined_data = Creator.combine_datasets(relative_joint_data, velocity_data, joint_bones_data, image_data,
                                             joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/19_Combined_Data")

    combined_data = Creator.create_dummy_dataset(combined_data, 
                                                 output_name="./Code/Datasets/Joint_Data/" + str(folder) + "/20_Combined_Data_Noise")

    

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
def load_datasets(types, folder):
    datasets = []
    print("loading datasets...")
        
    for i, t in enumerate(types):
        print("loading dataset {} of {}. ".format(i + 1, len(types)), t)
        #Type 1: Normal, full dataset
        if t == 1:  
            #15.5 COMBINED DATASET
            datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/3_people',
                                                      '3_people.csv',
                                                  joint_connections=Render.joint_connections_n_head))
            
            #Experimental
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/2_people/raw/2_people_rel', '2_people_rel.csv',
            #                                       joint_connections=Render.joint_connections_n_head))
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/2_people/raw/2_people_vel', '2_people_vel.csv',
            #                                      joint_connections=Render.joint_connections_n_head))
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/2_people/raw/2_people_bone', '2_people_bone.csv',
            #                                       joint_connections=Render.joint_connections_n_head))
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

    #Accounting for extra original dataset not used in dummy case for training but only for testing
    if 5 in dataset_types:
        num_datasets -= 1

    print("number of datasets: ", num_datasets)
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
                                                                     k_fold=5, batch=batch_size, epochs=epochs, type=model_type, device = device)

    #Process and display results
    process_results(train_scores, val_scores, test_scores)

def create_regions_data(data, folder):
    regions_data_2 = Creator.create_2_regions_dataset(data,
                                                       joint_output="./Code/Datasets/Joint_Data/"  + str(folder)  + "/2_region", images=None)
    regions_data_5 = Creator.create_5_regions_dataset(data, 
                                                      joint_output="./Code/Datasets/Joint_Data/"  + str(folder)  + "/5_region", images=None)

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


def load_whole_dataset(folder_names, file_name):
    data = []
    for name in folder_names:
        print("loading: ", "./Code/Datasets/Joint_Data/" + str(name) + str(file_name) + "/raw/"+ str(file_name) + ".csv")
        abs_joint_data, _ = Utilities.process_data_input("./Code/Datasets/Joint_Data/" + str(name) + str(file_name) + "/raw/"+ str(file_name) + ".csv", None,
                                                                cols=Utilities.colnames_nohead, ignore_depth=False)
        data.append(abs_joint_data)
    return data

def change_to_ensemble_classes():
    joint_data, _ = Utilities.process_data_input("./Code/Datasets/Joint_Data/chris/20_Combined_Data_Noise/raw/20_Combined_Data_Noise.csv",
                                                            None, cols=Utilities.colnames_nohead, ignore_depth=False)
    
    Creator.convert_person_to_type(joint_data, joint_output="./Code/Datasets/Joint_Data/chris/21_Combined_Data_Noise_Ensemble")
        
def extract_ensemble_data():
    joint_data, _ = Utilities.process_data_input("./Code/Datasets/Joint_Data/big/2_people/raw/2_people.csv",
                                                            None, cols=Utilities.colnames_nohead, ignore_depth=False)
    print("data loaded")
    create_regions_data(joint_data, "big")
    print("region data completed")
    unfold_3s_dataset(joint_data, joint_output="./Code/Datasets/Joint_Data/big/2_people/raw/2_people")
    print("unfolding completed")

def process_multiple(folders):
    for folder in folders:
        process_data(folder)

def stitch_dataset(folder_names):
    file_name = '/20_Combined_Data_Noise'
    datasets = load_whole_dataset(folder_names, file_name)
    whole_dataset = datasets[0]
    current_instance = whole_dataset[-1][0]
    for i, dataset in enumerate(datasets):
        if i > 0:
            current_instance, whole_dataset = Creator.assign_person_number(whole_dataset, dataset, 
                                                                       "./Code/Datasets/Joint_Data/Big/" + str(i + 1) + "_people",
                                                                       i, current_instance)
    print("completed.")

if __name__ == '__main__':
    process_data("Ahmed")
    process_data("Amy")
    process_data("Anna")
    process_data("Bob")
    process_data("Cade")
    process_data("Emma")
    process_data("Erin")
    process_data("Pheobe")
    process_data("Scarlett")
    process_data("Sean G")
    process_data("Wanok")

    #Done grant 100%, elisa 83%, sean c 98%, chris,

    #Assign person numbers and uniform instance counts:
    folder_names = ['Ahmed', 'Amy', 'Anna', 'Bob', 'Cade', 'Chris', 'Elisa', 'Grant', 'Emma',
                    'Erin', 'Pheobe', 'Scarlett', 'Sean G', 'Sean C' 'Wanok']
    
    stitch_dataset(folder_names=folder_names)
    
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
    
    #Grant results not great, sean c mid 70s, everyone else good

    #Do all together excluding grant and sean c then with

    #Do leave one-out

    #Do 3s vs 1s, 2 region and 5 region

    #Test best one with and without punishment

    #Test best one with and without skeleton subtraction

    #Get results on freeze or not, obstacle or not and person

    #Make ensemble classifier for all classes at once



    #Look into ST-GCN implementation to upload dataset to to see if ST-AGCN is better
    
    #Put into SVM and KNN for standard comparisons





    