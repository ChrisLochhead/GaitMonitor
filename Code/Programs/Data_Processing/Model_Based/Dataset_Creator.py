from Utilities import save_dataset

def create_abs_dataset():
    pass

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