
from Data_Processing.Model_Based.DatasetBuilder.demo import *
from Data_Processing.Model_Based.DatasetBuilder.hcf import create_hcf_dataset 
from Data_Processing.Model_Based.DatasetBuilder.render import * 
from Data_Processing.Model_Based.DatasetBuilder.data_correction import correct_joints_data, apply_joint_occlusion, normalize_joint_scales
from Data_Processing.Model_Based.DatasetBuilder.utilities import load, load_images

def main():

    #Process data from EDA into perfect form
    #correct_joints_data("./Images", "./EDA/gait_dataset_pixels.csv", save=True, pixels=True)
    #Test:
    #print("Correction processing sucessfully completed, testing resulting images and joints...")
    #load_and_overlay_joints(directory="./EDA/Finished_Data/Images/", joint_file="./EDA/Finished_Data/pixel_data_absolute.csv", ignore_depth=False, plot_3D=True)

    #Visualize metre's images to make sure they are proportional, especially regarding depth.
    #Get raw images and corresponding joint info
    #joint_data = load("./EDA/Finished_Data/metres_data_absolute.csv")
    #image_data = load_images("./EDA/Finished_Data/Images/")
    #rel_joint_data = load("./EDA/Finished_Data/pixel_data_absolute.csv")
    #image_iter = 0
    #for j in joint_data:
    #    #render_joints(image_data[image_iter], rel_joint_data[image_iter], delay=True)
    #    plot3D_joints(j, pixel = False)
    #    image_iter += 1


    joint_data = load("../EDA/Finished_Data/MPI_pixels_omit.csv")
    image_data = load_images("../EDA/Finished_Data/Images/")
    normalize_joint_scales(joint_data, image_data)

    #Experimental creating hand crafted features
    #create_hcf_dataset("../EDA/Finished_Data/pixel_data_absolute.csv", "../EDA/Finished_Data/pixel_data_relative.csv", \
    #                    "../EDA/Finished_Data/pixel_velocity_absolute.csv", "../EDA/Finished_Data/Images")

    #Create dataset with chest joints
    #create_dataset_with_chestpoint("./EDA/gait_dataset_pixels.csv", "./Images")
    
    #Demonstrate occlusion fixing
    #apply_joint_occlusion("./EDA/gait_dataset_pixels.csv", save = True, debug=True)

    #Draw calculated velocities
    #run_velocity_debugger("./EDA/Finished_Data/Images/", "./EDA/Finished_Data/pixel_data_relative.csv", save= True, debug=False)

    #run_images("./Images", exclude_2D=False, start_point=0)

    #run_depth_sample("./DepthExamples", "depth_examples.csv")
    #run_video()
if __name__ == '__main__':
    #Main menu
    main()