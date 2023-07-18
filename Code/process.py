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
    #                                              image_file="./Code/Datasets/" + str(folder) + "/Full_Dataset/",
    #                                              joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/2_Absolute_Data(empty frames removed)",
    #                                              image_output="./Code/Datasets/" + str(folder) + "/2_Empty Frames Removed/")

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
    
    abs_norm_data = Creator.normalize_values(abs_joint_data, 
                                             joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/3_Absolute_Data(normed)")
    print("\nStage 3: ")
    #render_joints_series(image_data, abs_joint_data, size=10)

    #Normalize outliers
    abs_joint_data = Creator.create_normalized_dataset(abs_joint_data, image_data, 
                                                   joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/4_Absolute_Data(normalized)")
    print("\nStage 4:")
    #render_joints_series(image_data, abs_joint_data, size=10)

    abs_joint_data = Creator.append_midhip(abs_joint_data, image_data, 
                                                   joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/4.5_Absolute_Data(midhip)")
     
    print("\nStage 4.5:")
    #render_joints_series(image_data, abs_joint_data, size=10)
    #Change format of pre-scale to list of arrays instead of list of lists
    pre_scale = abs_joint_data

    #Normalize size (use absolute dataset)
    abs_joint_data = Creator.create_scaled_dataset(abs_joint_data, image_data,
                                               joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/5_Absolute_Data(scaled)")

    print("\nStage 5: ")
    #render_joints_series(image_data, abs_joint_data, size=10)

    #Create relative dataset
    relative_joint_data = Creator.create_relative_dataset(abs_joint_data, image_data,
                                                 joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/6_Relative_Data(relative)")

    fake_data = Creator.create_dummy_dataset(relative_joint_data, output_name="./Code/Datasets/Joint_Data/" + str(folder) + "/0_Dummy_Data")
    
    print("\nStage 6:")
    #render_joints_series("None", relative_joint_data, size=5, plot_3D=True, x_rot = -90, y_rot = 180)

    #Flip joints so all facing one direction
    flipped_joint_data = Creator.create_flipped_joint_dataset(relative_joint_data, abs_joint_data, image_data,
                                                               joint_output = "./Code/Datasets/Joint_Data/" + str(folder) + "/7_Relative_Data(flipped)") 

    print("\nStage 7:")
    #render_joints_series("None", flipped_joint_data, size=5, plot_3D=True, x_rot = -90, y_rot = 180)

    #Create velocity dataset
    velocity_data = Creator.create_velocity_dataset(pre_scale, image_data, 
                                                    joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/8_Velocity_Data(velocity)")
    print("\nStage 8:")
    #render_velocity_series(abs_joint_data, velocity_data, image_data, size=20)

    flipped_velocity_data = Creator.create_flipped_joint_dataset(velocity_data, abs_joint_data, image_data,
                                                               joint_output = "./Code/Datasets/Joint_Data/" + str(folder) + "/9_Velocity_Data(flipped)")  
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
    print("\nStage 11:", pre_scale[0])
    #Create HCF dataset
    hcf_data = Creator.create_hcf_dataset(pre_scale, abs_joint_data, relative_joint_data, velocity_data, image_data, 
                               joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13_HCF_Data")
    
    hcf_data_normed = Creator.normalize_values(hcf_data, 
                                             joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/13.5_HCF_Data(normed)", hcf=True)
    
    print("\nStage 12:")
    #Create ground truth comparison set
    print("data going into gait cycle extractor: ", pre_scale[0])
    ground_truths = Creator.create_ground_truth_dataset(pre_scale, abs_joint_data, relative_joint_data, velocity_data, hcf_data, image_data,
                                                        joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/14_Ground_Truth_Data")
    
    print("\nStage 13:")
    #Combine datasets (relative, velocity, joint angles, regions)
    combined_data = Creator.combine_datasets(relative_joint_data, velocity_data, joint_bones_data, image_data,
                                             joints_output="./Code/Datasets/Joint_Data/" + str(folder) + "/15_Combined_Data")
    
    combined_norm_data = Creator.normalize_values(combined_data, 
                                             joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/15.5_Combined_Data(normed)")

    print("\nStage 14:")
    #Create regions data of combined data
    top_region_dataset, bottom_region_dataset = Creator.create_2_regions_dataset(combined_norm_data, #CHANGE BACK TO ABS_JOINT_DATA
                                                                                 joint_output="./Code/Datasets/Joint_Data/" + str(folder) + "/16_Combined_Data_2Region",
                                                                                 images = image_data)
    regions_data = Creator.create_5_regions_dataset(combined_data,
                                                    "./Code/Datasets/Joint_Data/" + str(folder) + "/17_Combined_Data_5Region",
                                                      image_data)
#########################################################################################################################

def load_2_region_data(cycle, preset = None, padding = False, folder = "Chris"):
    tmp = copy.deepcopy(cycle)

    top_region = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/16_Combined_Data_2Region_top',
                                        '16_Combined_Data_2Region_top.csv',
                                        joint_connections=Render.joint_connections_m_hip, cycles=True, cycle_preset=tmp, padding=padding)
    
    bottom_region = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/16_Combined_Data_2Region_bottom',
                                        '16_Combined_Data_2Region_bottom.csv',
                                        joint_connections=Render.joint_connections_m_hip, cycles=True, cycle_preset=tmp, padding=padding)
        
    if cycle:
        top_cycles, bottom_cycles = preset.split_cycles()
        top_region.data_cycles = top_cycles
        bottom_region.data_cycles = bottom_cycles
    
    return top_region, bottom_region

def load_5_region_data(cycles, preset = None, padding = False, folder = "Chris",):
    left_leg = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/17_Combined_Data_5Regionl_leg',
                                              '17_Combined_Data_5Regionl_leg.csv',
                                              joint_connections=Render.limb_connections, cycles=True, cycle_preset = copy.deepcopy(cycles), padding=padding)
    
    right_leg = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/17_Combined_Data_5Regionr_leg',
                                           '17_Combined_Data_5Regionr_leg.csv',
                                            joint_connections=Render.limb_connections, cycles=True, cycle_preset = copy.deepcopy(cycles), padding=padding)
    left_arm = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/17_Combined_Data_5Regionl_arm',
                                              '17_Combined_Data_5Regionl_arm.csv',
                                              joint_connections=Render.limb_connections, cycles=True, cycle_preset = copy.deepcopy(cycles), padding=padding)
    
    right_arm = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/17_Combined_Data_5Regionr_arm',
                                           '17_Combined_Data_5Regionr_arm.csv',
                                            joint_connections=Render.limb_connections, cycles=True, cycle_preset = copy.deepcopy(cycles), padding=padding)   
    head_data = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/17_Combined_Data_5Regionhead',
                                              '17_Combined_Data_5Regionhead.csv',
                                              joint_connections=Render.head_joint_connections, cycles=True, cycle_preset= copy.deepcopy(cycles), padding=padding)
    

    if cycles:
        l_leg, r_leg, l_arm, r_arm, head = preset.split_cycles(split_type=5)
        left_leg.data_cycles = l_leg
        right_leg.data_cycles = r_leg
        left_arm.data_cycles = l_arm
        right_arm.data_cycles = r_arm
        head_data.data_cycles = head

    return left_leg, right_leg, left_arm, right_arm, head_data

#Types are 1 = normal, 2 = HCF, 3 = 2 region, 4 = 5 region, 5 = Dummy. Pass types as an array of type numbers, always put hcf (2) at the END if including. If including HCF, you MUST include
#as cycles = True
def load_datasets(types, cycles, padding, folder):
    datasets = []
    print("loading datasets...")
    #Need gait cycle preset if gait cycles being used for single datapoints
    if cycles:
        gait_cycles_data = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/4.5_Absolute_Data(midhip)', '4.5_Absolute_Data(midhip).csv',
                                            joint_connections=Render.joint_connections_m_hip, cycles=True, padding=padding)

        full_region = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/15.5_Combined_Data(normed)', '15.5_Combined_Data(normed).csv',
                                                joint_connections=Render.joint_connections_m_hip, cycles=True,
                                                  cycle_preset=copy.deepcopy(gait_cycles_data.cycle_indices), padding=padding)
        
    for i, t in enumerate(types):
        print("loading dataset {} of {}. ".format(i + 1, len(types)), t)
        #Type 1: Normal, full dataset
        if t == 1 and cycles:  
            #15.5 COMBINED DATASET
            datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/15.5_Combined_Data(normed)', '15.5_Combined_Data(normed).csv',
                                                    joint_connections=Render.joint_connections_m_hip, cycles=True,
                                                      cycle_preset=copy.deepcopy(gait_cycles_data.cycle_indices), padding=padding))
            
            #7 co-ordinates on their own
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/7_Relative_Data(flipped)', '7_Relative_Data(flipped).csv',
            #                                        joint_connections=Render.joint_connections_m_hip, cycles=True,
            #                                          cycle_preset=copy.deepcopy(gait_cycles_data.cycle_indices), padding=padding))

            #12.75 no head coordinates
            #headless_cycles = full_region.split_cycles(1)
            #datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/12_75_no_head_Data__decimated', '12_75_no_head_Data__decimated.csv',
            #                                        joint_connections=Render.joint_connections_no_head_m_hip, cycles=True,
            #                                          cycle_preset=copy.deepcopy(gait_cycles_data.cycle_indices), padding=padding))
            #
            #datasets[-1].data_cycles = headless_cycles
            
        elif t == 1 and cycles == False:
            dataset = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/15.5_Combined_Data(normed)', '15.5_Combined_Data(normed).csv',
                                        joint_connections=Render.joint_connections_m_hip, cycles=False, padding=padding)
            datasets.append(dataset)
        #Type 2: HCF dataset
        elif t == 2 and cycles:
            #This MUST have cycles, there's no non-cycles option
            dataset = Dataset_Obj.HCFDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/13.5_HCF_Data(normed)',
                                                    '13.5_HCF_Data(normed).csv', cycles=True, cycle_preset=copy.deepcopy(gait_cycles_data.cycle_indices))
            datasets.append(dataset)
        #Type 3: 2 region
        elif t == 3:
            top_region, bottom_region = load_2_region_data(gait_cycles_data.cycle_indices, full_region, folder=folder, padding=padding)
            datasets.append(top_region)
            datasets.append(bottom_region)
        #Type 4: 5 region
        elif t == 4:
            l_l, r_l, l_a, r_a, h = load_5_region_data(copy.deepcopy(gait_cycles_data.cycle_indices), full_region, folder=folder, padding=padding)
            datasets.append(l_l)
            datasets.append(r_l)
            datasets.append(l_a)
            datasets.append(r_a)
            datasets.append(h)
        elif t == 5:
            datasets.append(Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/' + str(folder) + '/0_Dummy_Data',
                                                      '0_Dummy_Data.csv',
                                                    joint_connections=Render.joint_connections_m_hip, cycles=True, padding=padding))
                        

    print("datasets loaded.")
    #Return requested datasets
    return datasets

def process_datasets(datasets, dataset_size):
    print("Processing data...")
    train_val_indices = random.sample(range(dataset_size), int(0.9 * dataset_size))
    test_indices = random.sample(set(range(dataset_size)) - set(train_val_indices), int(0.1 * dataset_size))

    #These regions will be the same for both datasets
    multi_input_train_val = []
    multi_input_test = []
    for i, dataset in enumerate(datasets):
        multi_input_train_val.append(dataset[train_val_indices])
        multi_input_test.append(dataset[test_indices])
    
    
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

def process_autoencoder():
    print(f"Torch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Torch geometric version: {torch_geometric.__version__}")

    #Upper and lower region dataset
    lower_dataset = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/WeightGait/16_Combined_Data_2Region_bottom/', '16_Combined_Data_2Region_bottom.csv',
                                              joint_connections=Render.bottom_joint_connection).shuffle()
    dataset = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/WeightGait/15.5_Combined_Data(normed)/', '15.5_Combined_Data(normed).csv',
                                       joint_connections=Render.joint_connections_m_hip).shuffle()
    upper_dataset = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/WeightGait/16_Combined_Data_2Region_top/', '16_Combined_Data_2Region_top.csv',
                                             joint_connections=Render.top_joint_connections).shuffle()

    print(dataset[0])
    print(upper_dataset[0])
    print(lower_dataset[0])

    #train_loader, val_loader, test_loader, whole = GT.create_dataloaders(dataset)

    encoded_datasets = {'normal' : [dataset, "15.5_Combined_Data(normed)" ], 
                        'upper' : [upper_dataset, "16_Combined_Data_2Region_top"],
                        'lower' : [lower_dataset, "16_Combined_Data_2Region_bottom"]}
    
    skeleton_sizes = [17, 10, 6]

    #torch.manual_seed(0)
    num_epochs = 100
    #Upper Region Dataset
    
    for index, (key, value) in enumerate(encoded_datasets.items()):
        if index > -1:
            vae = GAE.VariationalAutoencoder(dim_in=value[0].num_node_features, dim_h=128, latent_dims=3)
            print("dim in: ", value[0].num_node_features)
            optim = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-5) # originally 5
            vae.to(device)
            vae.eval()
            for epoch in range(num_epochs):
                train_loader, _, test_loader, whole = GT.create_dataloaders(value[0], batch_size=8)
                train_loss = GAE.train_epoch(vae,device,train_loader,optim)
                #Get embedding of entire dataset and save it
                val_loss = GAE.test_epoch(vae,device,whole, './Code/Datasets/Joint_Data/WeightGait/' + str(value[1]) + '/encoded/raw/encoded.csv', 
                                        skeleton_size = skeleton_sizes[index])
                print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))


    print("Autoencoding completing: Merging top and bottom datasets...")

    encoded_dataset = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/WeightGait/15.5_Combined_Data(normed)/encoded/', 'encoded.csv',
                                                joint_connections=Render.joint_connections_m_hip).shuffle()
    encoded_upper = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/WeightGait/16_Combined_Data_2Region_top/encoded/', 'encoded.csv',
                                              joint_connections=Render.top_joint_connections).shuffle()
    encoded_lower = Dataset_Obj.JointDataset('./Code/Datasets/Joint_Data/WeightGait/16_Combined_Data_2Region_bottom/encoded/', 'encoded.csv',
                                              joint_connections=Render.bottom_joint_connection).shuffle()
    
    print("all encoded datasets loaded sucessfully. ")
    
    #Combine
    #Read both in as csv, append together, resacve, load result and joint dataset
    upper_array  = Utilities.load('./Code/Datasets/Joint_Data/WeightGait/16_Combined_Data_2Region_top/encoded/raw/encoded.csv', metadata=True, 
                                  colnames = Utilities.colnames_top)
    lower_array = Utilities.load('./Code/Datasets/Joint_Data/WeightGait/16_Combined_Data_2Region_bottom/encoded/raw/encoded.csv', metadata=True,
                                 colnames = Utilities.colnames_bottom)

    concatenated_regions = []
    for i, row in enumerate(upper_array):
        print("row: ", type(row), len(row), len(lower_array[i]), type(lower_array[i]))
        concat_row = list(row)
        for j, low_arr in enumerate(lower_array[i]):
            if j > 2:
                concat_row.append(low_arr)
        concatenated_regions.append(concat_row)
        print("lens: ", len(row), len(lower_array[i]), len(concatenated_regions[i]))
    
    Utilities.save_dataset(concatenated_regions, './Code/Datasets/Joint_Data/WeightGait/18_encoded_concat_2region/raw/encoded_concat_2region.csv')
    print("Concatenation sucessful...")

def run_model(dataset_types, cycles, model_type, hcf, batch_size, epochs, folder, multi):

    #Padding only necessary if using ST-GCN for 2D temporal convolutions
    padding = True
    if model_type == "GAT":
        padding = False

    #Load the full dataset alongside HCF with gait cycles
    datasets = load_datasets(dataset_types, cycles, padding, folder)
    #Concatenate data dimensions for ST-GCN
    data_dims = []
    for dataset in datasets:
        data_pair = [dataset.num_features, dataset.num_node_features]
        data_dims.append(data_pair)

    #Process datasets by manually shuffling to account for cycles
    multi_input_train_val, multi_input_test = process_datasets(datasets, len(datasets[0]))

    print("\nCreating {} model with {} datasets: ".format(model_type, len(datasets)))
    if model_type == "GAT":
        if multi:
            model = gat.MultiInputGAT(dim_in=[d.num_node_features for d in datasets], dim_h=128, dim_out=3, hcf=hcf, n_inputs=len(datasets))
        else:
            model = gat.GAT(dim_in=datasets[0].num_node_features, dim_h=128, dim_out=3)
    elif model_type == "ST-AGCN":
        if multi:
            model = stgcn.MultiInputSTGACN(dim_in=[d.num_node_features for d in datasets], dim_h=32, num_classes=3, n_inputs=len(datasets),
                                        data_dims=data_dims, batch_size=batch_size, hcf=hcf,
                                        max_cycle=datasets[0].max_cycle, num_nodes_per_graph=datasets[0].num_nodes_per_graph)
        else:
            model = stgcn.STGACN(dim_in=datasets[0].num_node_features, num_classes=3,
                                        batch_size=batch_size, max_cycle = datasets[0].max_cycle,
                                         num_nodes_per_graph = datasets[0].num_nodes_per_graph)

    else:
        print("Invalid model type.")
        return
    model = model.to("cuda")

    #Run cross-validated training
    train_scores, val_scores, test_scores = graph_utils.cross_valid(model, multi_input_test, datasets=multi_input_train_val,
                                                                     k_fold=3, batch=batch_size, epochs=epochs, type=model_type)

    #Process and display results
    process_results(train_scores, val_scores, test_scores)

if __name__ == '__main__':
    process_data("WeightGait")
    #process_autoencoder()

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

    run_model(dataset_types= [1,3,2], cycles = True, model_type = "ST-AGCN", hcf=True,
               batch_size = 32, epochs = 150, folder="WeightGait", multi=True)
