#import init_directories
from Programs.Data_Processing.Model_Based.Demo import *
#from Programs.Data_Processing.Model_Based.Utilities import load, load_images, save_dataset
import Programs.Data_Processing.Model_Based.Dataset_Creator as Creator

def main():

    #Experimental creating hand crafted features
    #create_hcf_dataset("../EDA/Finished_Data/pixel_data_absolute.csv", "../EDA/Finished_Data/pixel_data_relative.csv", \
    #                    "../EDA/Finished_Data/pixel_velocity_absolute.csv", "../EDA/Finished_Data/Images")

############################################# PIPELINE ##################################################################

    #Extract joints from images
    #run_images("./Code/Datasets/Office Images_Chris", "./Code/Datasets/Office_Dataset/", exclude_2D=False, start_point=0)

    
    #Sort class labels (for 0ffice Images_chris this is 20-0, 20-1, 20-2)
    '''abs_joint_data = assign_class_labels(num_switches=20, num_classes=2, 
                                    joint_file="./Code/Datasets/Joint_Data/Office_Dataset/Absolute_Data.csv",
                                    joint_output="./Code/Datasets/Joint_Data/Office_Dataset/1_Absolute_Data(classes applied).csv")
    
    #Display first 2 instances of results 
    print("\nStage 1: ")
    render_joints_series("./Code/Datasets/Office Images_Chris/Raw Images", joints=joint_data,
                         size = 100, delay=True, use_depth=True)
'''
    #Remove empty frames
    abs_joint_data, image_data = Creator.process_empty_frames(joint_file="./Code/Datasets/Joint_Data/Office_Dataset/1_Absolute_Data(classes applied).csv",
                                                  image_file="./Code/Datasets/Office Images_Chris/Raw Images/",
                                                  joint_output="./Code/Datasets/Joint_Data/Office_Dataset/2_Absolute_Data(empty frames removed).csv",
                                                  image_output="./Code/Datasets/Office Images_Chris/2_Empty Frames Removed/")

    #Display first 2 instances of results
    print("\nStage 2: ")
    #render_joints_series(image_data, abs_joint_data, size=15)
    #render_joints_series(image_data, abs_joint_data, size=15, plot_3D=True)

    #Trim start and end frames where joints get confused by image borders
    abs_joint_data, image_data =Creator.process_trimmed_frames(abs_joint_data, image_data,
                                                        joint_output="./Code/Datasets/Joint_Data/Office_Dataset/3_Absolute_Data(trimmed instances).csv",
                                                        image_output="./Code/Datasets/Office Images_Chris/3_Trimmed Instances/", trim = 5)

    print("\nStage 3: ")
    #render_joints_series(image_data, abs_joint_data, size=10)

    #Normalize outliers
    abs_joint_data = Creator.create_normalized_dataset(abs_joint_data, image_data, 
                                                   joint_output="./Code/Datasets/Joint_Data/Office_Dataset/4_Absolute_Data(normalized).csv")
    print("\nStage 4:")
    #render_joints_series(image_data, abs_joint_data, size=10)

    #Normalize size (use absolute dataset)
    abs_joint_data = Creator.create_scaled_dataset(abs_joint_data, image_data,
                                               joint_output="./Code/Datasets/Joint_Data/Office_Dataset/5_Absolute_Data(scaled).csv")

    print("\nStage 5: ")
    #render_joints_series(image_data, scaled_joint_data, size=10)

    #Create relative dataset
    relative_joint_data = Creator.create_relative_dataset(abs_joint_data, image_data,
                                                 joint_output="./Code/Datasets/Joint_Data/Office_Dataset/6_Relative_Data(relative).csv")
    print("\nStage 6:")
    #render_joints_series("None", relative_joint_data, size=5, plot_3D=True, x_rot = -90, y_rot = 180)

    #Flip joints so all facing one direction
    flipped_joint_data = Creator.create_flipped_joint_dataset(relative_joint_data, abs_joint_data, image_data,
                                                               joint_output = "./Code/Datasets/Joint_Data/Office_Dataset/7_Relative_Data(flipped).csv")     
    print("\nStage 7:")
    #render_joints_series("None", flipped_joint_data, size=5, plot_3D=True, x_rot = -90, y_rot = 180)

    #Create velocity dataset
    velocity_data = Creator.create_velocity_dataset(abs_joint_data, image_data, 
                                                    joint_output="./Code/Datasets/Joint_Data/Office_Dataset/8_Velocity_Data(velocity).csv")

    print("\nStage 8:")
    render_velocity_series(abs_joint_data, velocity_data, image_data, size=20)

    flipped_velocity_data = Creator.create_flipped_joint_dataset(velocity_data, abs_joint_data, image_data,
                                                               joint_output = "./Code/Datasets/Joint_Data/Office_Dataset/9_Velocity_Data(flipped).csv")  
    
    print("\nStage 9: ")
    #Create joint angles data
    joint_angles_data = Creator.create_joint_angle_dataset(abs_joint_data, 
                                                            joint_output="./Code/Datasets/Joint_Data/Office_Dataset/10_Angles_Data(integrated).csv")

    alt_joint_angles = Creator.create_disjointed_angle_dataset(abs_joint_data,
                                                                joint_output="./Code/Datasets/Joint_Data/Office_Dataset/11_Angles_Data(disjointed).csv")
    #Create regions data
    top_region_dataset, bottom_region_dataset = Creator.create_2_regions_dataset(abs_joint_data, 
                                                                                 joint_output="./Code/Datasets/Joint_Data/Office_Dataset/12_2_Region_Data")
    regions_data = Creator.create_5_regions_dataset(abs_joint_data,
                                                    "./Code/Datasets/Joint_Data/Office_Dataset/13_5_Data_",
                                                      image_data)
    print("\nStage 10:")
    #Create HCF dataset
    hcf_data = Creator.create_hcf_dataset(abs_joint_data, relative_joint_data, velocity_data, image_data, 
                               joints_output="./Code/Datasets/Joint_Data/Office_Dataset/14_HCF_Data")

    print("\nStage 11:")
    #Create ground truth comparison set
    ground_truths = Creator.create_ground_truth_dataset(abs_joint_data, relative_joint_data, velocity_data, hcf_data, image_data,
                                                        joints_output="./Code/Datasets/Joint_Data/Office_Dataset/15_Ground_Truth_Data")
    
    print("\nStage 12:")
    #Combine datasets (relative, velocity, joint angles, regions)
    combined_data = Creator.combine_datasets(relative_joint_data, velocity_data, joint_angles_data, image_data,
                                             joints_output="./Code/Datasets/Joint_Data/Office_Dataset/16_Combined_Data")

#########################################################################################################################



if __name__ == '__main__':
    #Main menu
    main()