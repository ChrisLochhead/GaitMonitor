from Programs.Data_Processing.Model_Based.Utilities import save_dataset, load, load_images, save_images
from Programs.Data_Processing.Model_Based.Data_Correction import correct_joints_data, normalize_joint_scales, remove_empty_frames, apply_class_labels

def assign_class_labels(num_switches, num_classes, joint_file, joint_output):
    joint_data = load(joint_file)
    joint_data = apply_class_labels(num_switches, num_classes, joint_data)
    save_dataset(joint_data, joint_output)
    return joint_data

def process_empty_frames(joint_file, image_file, joint_output, image_output):
    joint_data = load(joint_file)
    image_data = load_images(image_file)
    print("loading in data: ", len(joint_data), len(image_data))
    joint_data, image_data = remove_empty_frames(joint_data, image_data)
    print("full frame count: ", len(image_data), len(joint_data))
    save_dataset(joint_data, joint_output)
    save_images(joint_data, image_data, image_output)

    return joint_data, image_data

def create_relative_dataset(abs_data):
    rel_data = []
    for i, joints in enumerate(abs_data):
        rel_row = []
        for j, coord in enumerate(joints):
            #Ignore metadata
            origin = joints[3]
            if j < 3:
                rel_row.append(coord)
            elif j == 3:
                #Origin point, set to 0
                rel_row.append([0,0,0])
            else:
                print("origin: ", origin, coord)
                #Regular coord, append relative to origin
                rel_row.append([abs(origin[0] - coord[0]),
                                abs(origin[1] - coord[1]),
                                abs(origin[2] - coord[2])])

        rel_data.append(rel_row)
    
    save_dataset(rel_data, "../../../Datasets/Joint_Data/New/Relative_Data.csv")

def create_normalized_dataset():
    pass

def create_velocity_dataset():
    pass

def create_joint_angle_dataset():
    pass

def create_hcf_dataset():
    pass

def create_regions_dataset():
    pass