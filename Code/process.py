#import init_directories
from Programs.Data_Processing.Model_Based.Demo import *
#from Programs.Data_Processing.Model_Based.Utilities import load, load_images, save_dataset
import Programs.Data_Processing.Model_Based.Dataset_Creator as Creator
import Programs.Machine_Learning.Model_Based.AutoEncoder.GAE as GAE
import Programs.Machine_Learning.Model_Based.GCN.Dataset_Obj as Dataset_Obj
import Programs.Machine_Learning.Model_Based.GCN.Ground_Truths as GT
import Programs.Data_Processing.Model_Based.Render as Render
import torch
import torch_geometric
import random
import Programs.Machine_Learning.Model_Based.GCN.GAT as gat
import Programs.Machine_Learning.Model_Based.GCN.STAGCN as stgcn
import Programs.Machine_Learning.Model_Based.GCN.Utilities as graph_utils
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_data(folder = "Chris"):

############################################# PIPELINE ##################################################################

    #Extract joints from images
    #run_images("./Code/Datasets/" + str(folder) + "/Full_Dataset", out_folder="./Code/Datasets/Joint_Data/" + str(folder) + "/", exclude_2D=False, 
    #           start_point=0)

    
    #Sort class labels (for 0ffice Images_chris this is 20-0, 20-1, 20-2)
    #abs_joint_data = Creator.assign_class_labels(num_switches=20, num_classes=2, 
    #                                joint_file="./Code/Datasets/Joint_Data/WeightGait/Absolute_Data.csv",
    #                                joint_output="./Code/Datasets/Joint_Data/WeightGait/1_Absolute_Data(classes applied).csv")
    
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
    print("lens: ", len(abs_joint_data), len(image_data))
    
    #abs_norm_data = Creator.normalize_values(abs_joint_data, 
    #                                         joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/3_Absolute_Data(normed)")
    
    abs_joint_data = Creator.new_normalize_values(abs_joint_data, "./Code/Datasets/Joint_Data/" + str(folder) + "/3_Absolute_Data(normed)", 3)

    

    print("\nStage 3: ")
    #render_joints_series(image_data, abs_joint_data, size=10)

    #Normalize outliers
    #SIMPLIFY
    #abs_joint_data = Creator.create_normalized_dataset(abs_joint_data, image_data, 
    #                                               joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/4_Absolute_Data(normalized)")
    print("\nStage 4:")
    #render_joints_series(image_data, abs_joint_data, size=10)

    abs_joint_data = Creator.append_midhip(abs_joint_data, image_data, 
                                                   joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/4.5_Absolute_Data(midhip)")
     
    print("\nStage 4.5:")
    #render_joints_series(image_data, abs_joint_data, size=10)
    #Change format of pre-scale to list of arrays instead of list of lists
    #pre_scale = abs_joint_data

    #Normalize size (use absolute dataset)
    #abs_joint_data = Creator.create_scaled_dataset(abs_joint_data, image_data,
    #                                           joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/5_Absolute_Data(scaled)")

    print("\nStage 5: ")
    #render_joints_series(image_data, abs_joint_data, size=10)

    #Create relative dataset
    relative_joint_data = Creator.create_relative_dataset(abs_joint_data, image_data,
                                                 joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/6_Relative_Data(relative)")
    
    print("\nStage 6:")
    #render_joints_series("None", relative_joint_data, size=5, plot_3D=True, x_rot = -90, y_rot = 180)

    #Flip joints so all facing one direction
    #abs_joint_data = Creator.create_flipped_joint_dataset(relative_joint_data, abs_joint_data, image_data,
    #                                                           joint_output = "./Code/Datasets/Joint_Data/" + str(folder) + "/7_Relative_Data(flipped)") 

    print("\nStage 7:")
    #render_joints_series("None", flipped_joint_data, size=5, plot_3D=True, x_rot = -90, y_rot = 180)

    #Create velocity dataset
    velocity_data = Creator.create_velocity_dataset(abs_joint_data, image_data, 
                                                    joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/8_Velocity_Data(velocity)")
    print("\nStage 8:")
    #render_velocity_series(abs_joint_data, velocity_data, image_data, size=20)

    #flipped_velocity_data = Creator.create_flipped_joint_dataset(velocity_data, abs_joint_data, image_data,
    #                                                           joint_output = "./Code/Datasets/Joint_Data/" + str(folder) + "/9_Velocity_Data(flipped)")  

    print("\nStage 9: ")
    #Create joint angles data
    joint_bones_data = Creator.create_bone_dataset(abs_joint_data, image_data, 
                                                   joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/10_Bones_Data(integrated)")
    
    #render_velocity_series(abs_joint_data, joint_bones_data, image_data, size=20)

    print("\nStage 10:")
    #Create regions data
    top_region_dataset, bottom_region_dataset = Creator.create_2_regions_dataset(abs_joint_data, #CHANGE BACK TO ABS_JOINT_DATA
                                                                                 joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/11_2_Region_Data",
                                                                                 images = image_data)
    regions_data = Creator.create_5_regions_dataset(abs_joint_data,
                                                    "./Code/Datasets/Joint_Data/" + str(folder) + "/12_5_Data_",
                                                      image_data)
    
    regions_data = Creator.create_decimated_dataset(abs_joint_data,
                                                "./Code/Datasets/Joint_Data/" + str(folder) + "/12_75_no_head_Data_",
                                                    image_data)
    #print("\nStage 11:", pre_scale[0])
    #Create HCF dataset
    #hcf_data = Creator.create_hcf_dataset(pre_scale, abs_joint_data, relative_joint_data, velocity_data, image_data, 
    #                           joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_HCF_Data")
    
    #hcf_data_normed = Creator.normalize_values(hcf_data, 
    #                                         joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13.5_HCF_Data(normed)", hcf=True)
    
    print("\nStage 12:")
    #Create ground truth comparison set
    #print("data going into gait cycle extractor: ", pre_scale[0])
    #ground_truths = Creator.create_ground_truth_dataset(pre_scale, abs_joint_data, relative_joint_data, velocity_data, hcf_data, image_data,
    #                                                    joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/14_Ground_Truth_Data")
    
    print("\nStage 13:")
    #Combine datasets (relative, velocity, joint angles, regions)
    combined_data = Creator.combine_datasets(relative_joint_data, velocity_data, joint_bones_data, image_data,
                                             joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/15_Combined_Data")
    
    #combined_norm_data = Creator.normalize_values(combined_data, 
    #                                         joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/15.5_Combined_Data(normed)")
    
    #combined_norm_data = Creator.new_normalize_values(combined_data, 
    #                                         "./Code/Datasets/Joint_Data/" + str(folder) + "/15.5_Combined_Data(normed)", 9)   
    combined_norm_data = combined_data
    fake_data = Creator.create_dummy_dataset(combined_norm_data, output_name="./Code/Datasets/Joint_Data/" + str(folder) + "/0_Dummy_Data_15.5")
    
    print("\nStage 14:")
    #Create regions data of combined data
    top_region_dataset, bottom_region_dataset = Creator.create_2_regions_dataset(combined_norm_data, #CHANGE BACK TO ABS_JOINT_DATA
                                                                                 joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/16_Combined_Data_2Region",
                                                                                 images = image_data)
    regions_data = Creator.create_5_regions_dataset(combined_norm_data,
                                                    "./Code/Datasets/Joint_Data/" + str(folder) + "/17_Combined_Data_5Region",
                                                      image_data)
    
    fused_data = Creator.create_fused_dataset(combined_norm_data, "./Code/Datasets/Joint_Data/" + str(folder) + "/18_Combined_Data_Fused")

    precise_data = Creator.normal_examples_only(combined_norm_data, "./Code/Datasets/Joint_Data/" + str(folder) + "/19_Normal_Only" )
    
    if folder == "WeightGait":
        full_data, _ = Utilities.process_data_input("./Code/Datasets/Joint_Data/" + str(folder) + "/15_Combined_Data/raw/15_Combined_Data.csv",
                                                              "./Code/Datasets/" + str(folder) + "/15_Combined_Data/", cols=Utilities.colnames_midhip, ignore_depth=False)
        two_person_data = Creator.create_n_size_dataset(full_data, "./Code/Datasets/Joint_Data/" + str(folder) + "/20_1_People", [1])
        two_person_data = Creator.create_n_size_dataset(full_data, "./Code/Datasets/Joint_Data/" + str(folder) + "/20_2_People", [0,1])
        two_person_data = Creator.create_n_size_dataset(full_data, "./Code/Datasets/Joint_Data/" + str(folder) + "/20_3_People", [0,1,2])
        two_person_data = Creator.create_n_size_dataset(full_data, "./Code/Datasets/Joint_Data/" + str(folder) + "/20_4_People", [0,3,5,6])
        two_person_data = Creator.create_n_size_dataset(full_data, "./Code/Datasets/Joint_Data/" + str(folder) + "/20_Cade", [2])
        two_person_data = Creator.create_n_size_dataset(full_data, "./Code/Datasets/Joint_Data/" + str(folder) + "/20_Elisa", [4])
        two_person_data = Creator.create_n_size_dataset(full_data, "./Code/Datasets/Joint_Data/" + str(folder) + "/20_Longfei", [5])
        two_person_data = Creator.create_n_size_dataset(full_data, "./Code/Datasets/Joint_Data/" + str(folder) + "/20_Pheobe", [6])
        two_person_data = Creator.create_n_size_dataset(full_data, "./Code/Datasets/Joint_Data/" + str(folder) + "/20_SeanC", [7])
        two_person_data = Creator.create_n_size_dataset(full_data, "./Code/Datasets/Joint_Data/" + str(folder) + "/20_SeanG", [8])
#########################################################################################################################

def load_2_region_data(folder = "Chris"):
    top_region = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/16_Combined_Data_2Region_top',
                                        '16_Combined_Data_2Region_top.csv',
                                        joint_connections=Render.joint_connections_m_hip, cycles=True)
    
    bottom_region = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/16_Combined_Data_2Region_bottom',
                                        '16_Combined_Data_2Region_bottom.csv',
                                        joint_connections=Render.joint_connections_m_hip, cycles=True)
           
    return top_region, bottom_region

def load_5_region_data(folder = "Chris"):
    left_leg = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/17_Combined_Data_5Regionl_leg',
                                              '17_Combined_Data_5Regionl_leg.csv',
                                              joint_connections=Render.limb_connections, cycles=True)
    
    right_leg = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/17_Combined_Data_5Regionr_leg',
                                           '17_Combined_Data_5Regionr_leg.csv',
                                            joint_connections=Render.limb_connections, cycles=True)
    left_arm = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/17_Combined_Data_5Regionl_arm',
                                              '17_Combined_Data_5Regionl_arm.csv',
                                              joint_connections=Render.limb_connections, cycles=True)
    
    right_arm = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/17_Combined_Data_5Regionr_arm',
                                           '17_Combined_Data_5Regionr_arm.csv',
                                            joint_connections=Render.limb_connections, cycles=True)   
    head_data = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/17_Combined_Data_5Regionhead',
                                              '17_Combined_Data_5Regionhead.csv',
                                              joint_connections=Render.head_joint_connections, cycles=True)

    return left_leg, right_leg, left_arm, right_arm, head_data

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
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/15_Combined_Data', '15_Combined_Data.csv',
            #                                        joint_connections=Render.joint_connections_m_hip, cycles=True))
            
            #20 multi person DATASET
            datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/20_4_people', '20_4_people.csv',
                                                   joint_connections=Render.joint_connections_m_hip, cycles=True))
            
            #7 co-ordinates on their own
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/7_Relative_Data(flipped)', '7_Relative_Data(flipped).csv',
            #                                        joint_connections=Render.joint_connections_m_hip, cycles=True))

            #19 simplified dataset
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/19_Normal_Only', '19_Normal_Only.csv',
            #                                        joint_connections=Render.joint_connections_no_head_m_hip, cycles=True))
            
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/0_Dummy_Data_15.5',
            #                                          '0_Dummy_Data_15.5.csv',
             #                                       joint_connections=Render.joint_connections_m_hip, cycles=True))
            
        elif t == 1:
            dataset = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/15.5_Combined_Data(normed)', '15.5_Combined_Data(normed).csv',
                                        joint_connections=Render.joint_connections_m_hip, cycles=False)
            datasets.append(dataset)
        #Type 2: HCF dataset
        elif t == 2:
            #This MUST have cycles, there's no non-cycles option
            dataset = Dataset_Obj.HCFDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/13.5_HCF_Data(normed)',
                                                    '13.5_HCF_Data(normed).csv', cycles=True)
            datasets.append(dataset)
        #Type 3: 2 region
        elif t == 3:
            top_region, bottom_region = load_2_region_data(folder)
            datasets.append(top_region)
            datasets.append(bottom_region)
        #Type 4: 5 region
        elif t == 4:
            l_l, r_l, l_a, r_a, h = load_5_region_data(folder)
            datasets.append(l_l)
            datasets.append(r_l)
            datasets.append(l_a)
            datasets.append(r_a)
            datasets.append(h)
        #Type 5: Dataset with dummy datapoints
        elif t == 5:
            #Dummy dataset
            datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/0_Dummy_Data_15.5',
                                                      '0_Dummy_Data_15.5.csv',
                                                    joint_connections=Render.joint_connections_m_hip, cycles=True))
            
            #15.5 COMBINED DATASET (original version of the dummy dataset)
            datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/15_Combined_Data', '15_Combined_Data.csv',
                                                    joint_connections=Render.joint_connections_m_hip, cycles=True))                        

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
        train_indices = random.sample(range(dataset_size), int(0.7 * dataset_size))
        print("original indices:", len(train_indices), train_indices )
        test_indices = random.sample(set(range(dataset_size)) - set(train_indices), int(0.3 * dataset_size))
        print("original test indices:", len(test_indices), test_indices )
        #train_indices, test_indices = get_balanced_samples(datasets[0])
        #done = 5/0
        train_indice_list.append(train_indices)
        test_indice_list.append(test_indices)

    #These regions will be the same for both datasets
    multi_input_train_val = []
    multi_input_test = []
    for i, dataset in enumerate(datasets):
        multi_input_train_val.append(dataset[train_indice_list[i]])
        multi_input_test.append(dataset[test_indice_list[i]])
    
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

def process_autoencoder(folder, num_epochs, batch_size):
    #load dataset
    dataset = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/15.5_Combined_Data(normed)', '15.5_Combined_Data(normed).csv',
                                            joint_connections=Render.joint_connections_m_hip, cycles=True)             

    name = "15.5_Combined_Data(normed)"

    vae = GAE.VariationalAutoencoder(dim_in=dataset.num_nodes_per_graph, dim_h=128, latent_dims=3, batch_size=batch_size, cycle_size=dataset.max_cycle)
    optim = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-5) # originally 5
    vae.to(device)
    vae.eval()
    for epoch in range(num_epochs):
        _, _, _, whole = GT.create_dataloaders(dataset, batch_size=batch_size)
        train_val_data, test_data = process_datasets([dataset])
        train_loader, val_loader, test_loader = graph_utils.cross_valid(None, test_data, datasets=train_val_data, make_loaders=True, batch=batch_size)

        train_loss = GAE.train_epoch(vae,device,train_loader[0],optim)
        #Get embedding of entire dataset and save it
        val_loss = GAE.test_epoch(vae,device,whole, './Code/Datasets/Joint_Data/WeightGait/' + str(name) + '/encoded/raw/encoded.csv', 
                                skeleton_size = 21)
        print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))


    print("Autoencoding complete")

    #Load autoencoded model (data will be 16 * 3 * 71)

    #Compress

def run_model(dataset_types, model_type, hcf, batch_size, epochs, folder, leave_one_out):

    #Load the full dataset alongside HCF with gait cycles
    datasets = load_datasets(dataset_types, folder)
    print("datasets here: ", datasets)
    #Concatenate data dimensions for ST-GCN
    data_dims = []
    for dataset in datasets:
        data_pair = [dataset.num_features, dataset.num_node_features]
        data_dims.append(data_pair)

    num_datasets = len(datasets)

    #Accounting for extra original dataset not used in dummy case for training but only for testing
    if 5 in dataset_types:
        num_datasets -= 1

    print("number of datasets: ", num_datasets)
    
    #Split classes by just making the last person the test set and the rest training and validation.
    if leave_one_out:
        multi_input_train_val, multi_input_test = graph_utils.split_data_by_person(datasets)
    else:
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
                                    max_cycle=datasets[0].max_cycle, num_nodes_per_graph=datasets[0].num_nodes_per_graph)
    else:
        print("Invalid model type.")
        return
    model = model.to("cuda")

    train_scores, val_scores, test_scores = graph_utils.cross_valid(model, multi_input_test, datasets=multi_input_train_val,
                                                                     k_fold=5, batch=batch_size, epochs=epochs, type=model_type)

    #Process and display results
    process_results(train_scores, val_scores, test_scores)

if __name__ == '__main__':
    #process_data("WeightGait")
    #process_autoencoder("Elisa", 100, 8)

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

    run_model(dataset_types= [1], model_type = "ST-AGCN", hcf=False,
           batch_size = 16, epochs = 100, folder="WeightGait", leave_one_out=False)

    #Full datset can overfit at 130 - 180 epochs at 8 batch size
    #Full dataset at 16 overfits at <100 epochs

    #Things to try:
    # try with 2, 3, 4 see where valid stops cause it can get 75% on Chris now, this is 88% validation on 100 epochs and 16 batch size
    #Full dataset one person only (person 0) : 61%
    #Bob = random (32%)
    #Cade = worse than random (25%)
    #Elisa = random (35%)
    #Longfei = decent (55%)
    #Pheobe = decent (62%)
    #Sean C = less than decent (40%)
    #Sean G = random (37%, 52%, 65%, 50%) 40 epochs: (62%, 50%)
    #Two people (0, 1) = random


    # try 1,2,3,4 with multiple datasets (need to implement code for that)
    #try 1000 epochs 
    #Find where it just stops working on valid, need bigger network probably (4 layers, 128,128,256,256 is what paper used
    # Try splitting data into 4 again
    # Test on telling people apart as the class
    # Test small dataset with noise for verification that it's good
