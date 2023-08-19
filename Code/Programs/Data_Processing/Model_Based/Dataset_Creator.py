import Programs.Data_Processing.Model_Based.Data_Correction as Data_Correction
import Programs.Data_Processing.Model_Based.Utilities as Utilities
import Programs.Data_Processing.Model_Based.Render as Render
import Programs.Data_Processing.Model_Based.HCF as hcf
from tqdm import tqdm
import math
import numpy as np
import copy 

from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

def average_coordinate(datapoint):
    #Aggregate the 9 values of every datapoint in the set, normalized
    aggregate = list(np.zeros(len(datapoint[0])))
    for data in datapoint:
        for i, coord in enumerate(data):
            aggregate[i] += (coord / len(datapoint[0]))
    
    return aggregate

def normal_examples_only(data, joint_output):
    normals = []
    current_instance = 0
    true_instance = 0
    for row in data:
        if row[3] == 0 and row[4] == 0 and row[2] < 3:
            if row[0] != current_instance:
                current_instance = row[0]
                true_instance += 1
            tmp = row
            tmp[0] = true_instance
            normals.append(tmp)
    
    Utilities.save_dataset(normals, joint_output)
    return normals

def create_n_size_dataset(data, joint_output, n):
    new_dataset = []
    for i, row in enumerate(data):
        if row[5] in n:
            print("discovered:", row[0], i, n)
            new_dataset.append(copy.deepcopy(row))

    Utilities.save_dataset(new_dataset, joint_output)


def create_fused_dataset(data, joint_output, meta = 6):
    fused_dataset = []
    for row in data:
        meta_data = row[0:meta]
        head_data = row[meta:meta+6]
        left_arm = [row[11], row[13], row[15]]
        right_arm = [row[12], row[14], row[16]]
        legs = row[17:]

        #Aggregate the head and arms
        head_agg = average_coordinate(head_data)
        l_agg = average_coordinate(left_arm)
        r_agg = average_coordinate(right_arm)

        #Stack them back together
        meta_data.append(head_agg)
        meta_data.append(l_agg)
        meta_data.append(r_agg)    

        for d in legs:
            meta_data.append(d)      

        fused_dataset.append(meta_data)

    Utilities.save_dataset(fused_dataset, joint_output)
    return meta_data

def normalize_hcf(data, joint_output):
    # create scaler
    scaler = StandardScaler()
    # fit scaler on data
    scaler.fit(data)
    # apply transform
    standardized = scaler.transform(data)
    Utilities.save_dataset(standardized, joint_output)
    return standardized

def new_normalize_values(data, joint_output, joint_size):
    # create scaler
    scaler = StandardScaler()
    # fit scaler on data

    print("data: ", len(data), len(data[0]))
    meta_data = []
    joint_info = []
    for row in data:
        meta_row = []
        joint_row = []
        for i, coord in enumerate(row):
            if i <= 5:
                meta_row.append(coord)
            else:
                joint_row.append(coord)

        meta_data.append(meta_row)
        joint_info.append(joint_row)
    
    print("lens: ", len(meta_data), len(meta_data[0]), len(joint_info), len(joint_info[0]))
            
    unravelled_joints = []
    for row in joint_info:
        unravelled_row = []
        for coords in row:
            for value in coords:
                unravelled_row.append(value)
        unravelled_joints.append(unravelled_row)

    print("lens:", len(unravelled_joints), len(unravelled_joints[0]))

    scaler.fit(unravelled_joints)

    # apply transform
    standardized = scaler.transform(unravelled_joints)

    print("should be the same:", len(standardized), len(standardized[0]))


    #Re-ravel
    ravelled = []
    for i, row in enumerate(standardized):
        ravelled_row = meta_data[i]
        full_coord = []
        for j, coord in enumerate(row):
            if j % joint_size == 0 and j != 0:
                ravelled_row.append(copy.deepcopy(full_coord))
                full_coord = [coord]
            else:
                full_coord.append(coord)
        ravelled_row.append(copy.deepcopy(full_coord))
        ravelled.append(ravelled_row)
    
    print("final ravel: ", len(ravelled), len(ravelled[0]))

    Utilities.save_dataset(ravelled, joint_output)
    return ravelled

def normalize_values(data, joint_output, hcf = False, meta = 5):
    if hcf:
        data, _ = Utilities.process_data_input(data, None, cols=Utilities.hcf_colnames)
    else:
        data, _ = Utilities.process_data_input(data, None, cols=Utilities.colnames)
    print("data length: ", len(data[0]))
    meta_data = []
    joint_data = []

    for i, column in enumerate(zip(*data)): 
        if i <= meta: 
            meta_data.append(column)
        else:
            #print("column before: ", len(column), column)
            if hcf:
                normed_matrix = Utilities.normalize_1D(column)
                #normed_matrix = normalize(column, axis=0, norm='l2')  
            else:
                normed_matrix = normalize(column, axis=1, norm='l2')  
            #print("column after: ", len(normed_matrix), len(normed_matrix[0]))
            joint_data.append(normed_matrix)
            #print("lens: ", len(normed_matrix), len(column), len(column[0]))

    print("joint data: ", len(joint_data[0]))
    transposed_joints = []
    for index, j in enumerate(zip(*joint_data)):
        transposed_joints.append(j)

    joint_data = transposed_joints
    #print("joint data info: ", len(joint_data), len(joint_data[0]), len(joint_data[0][0]))
    meta_data = np.transpose(meta_data)
    print("meta data: ", len(meta_data[0]))
    final = []
    #print("meta data shape: ", len(meta_data), len(meta_data[1]))
    for j, col in enumerate(meta_data):
        final_row = list(col)
        for coords in joint_data[j]:
            #Fix the formatting before saving it 
            if hcf == False:
                refactored_coords = []
                for c in coords:
                    refactored_coords.append(c)
                final_row.append(refactored_coords)
            else:
                final_row.append(coords)

        final.append(final_row)
   

    print("joints: ", len(final[0]))
    #print("joints dimensions: ", len(final[0]))
    #print(len(final[0][4]))
    if hcf:
        print("saving as this: ", len(final), len(final[0]))
        Utilities.save_dataset(final, joint_output, colnames=Utilities.hcf_colnames) 
    else:
        Utilities.save_dataset(final, joint_output)
    return final

def avg_coord(data):
    total_coord = data[0]
    for i, d in enumerate(data):
        if i > 0:
            for j, coord in enumerate(d):
                total_coord[j] += coord
    
    for i, coord in enumerate(total_coord):
        total_coord[i] /= len(data)

    return total_coord

def combine_datasets(rel_data, vel_data, angle_data, images, joints_output, meta = 5):
    print("Combining datasets...")
    rel_data, images = Utilities.process_data_input(rel_data, images)
    vel_data, _ = Utilities.process_data_input(vel_data, None)
    angle_data, _ = Utilities.process_data_input(angle_data, None)
    print("combine lens: ", len(rel_data), len(vel_data), len(angle_data), len(rel_data[0]), len(vel_data[0]), len(angle_data[0]))
    combined_dataset = []
    for i, row in enumerate(tqdm(rel_data)):
        #Metadata is the same as usual
        combined_row = row[0:meta + 1]
        for j, joint in enumerate(row):
            if j > meta:
                if j == meta + 1:
                    avg_joint = avg_coord(row[meta + 2: meta + 9])
                    avg_vel = avg_coord(vel_data[i][meta + 2: meta + 9])
                    avg_ang = avg_coord(angle_data[i][meta + 2: meta + 9])

                    combined_row.append([avg_joint[0], avg_joint[1], avg_joint[2],
                    avg_vel[0], avg_vel[1], avg_vel[2], 
                    avg_ang[0], avg_ang[1], avg_ang[2] ])
                elif j > 10:
                #print("row before: ", combined_row[0:5])
                #print("lens: ", len(joint), len(vel_data[i][j]), len(angle_data[i][j]))
                    combined_row.append([joint[0], joint[1], joint[2],
                                        vel_data[i][j][0], vel_data[i][j][1], vel_data[i][j][2], 
                                        angle_data[i][j][0], angle_data[i][j][1], angle_data[i][j][2] ])
                #print("row after: ", combined_row[0:5])
        
        #print("final combined row: ", combined_row[0:5])
        combined_dataset.append(combined_row)
    
    print("Completing combined dataset.")
    Utilities.save_dataset(combined_dataset, joints_output)
    return combined_dataset

def combined_ground_truth_dataset(hcf, ground_truths, joints_output):
    hcf, _ = Utilities.process_data_input(hcf, None)
    ground_truths, _ = Utilities.process_data_input(ground_truths, None)
    combined_data = []

    print("enumerating hcf: ", len(hcf))
    for i, row in enumerate(hcf):
        print("first type {}, row type: {}".format(type(ground_truths[i]), type(row)))
        concat = row + ground_truths[i]
        print("lens: ", len(concat), len(row), len(ground_truths[i]))
        combined_data.append(row + ground_truths[i])

    Utilities.save_dataset(combined_data, joints_output, colnames=Utilities.combined_colnames)


def create_ground_truth_dataset(pre_abs_data, abs_data, rel_data, vel_data, hcf_data, images, joints_output, meta = 5):
    #Intuition: just averaging data will do little to outline potential discriminating data, especially with absolute data, rel or velocities.
    #Instead 
    print("Creating ground truth datasets...")

    abs_data, images = Utilities.process_data_input(abs_data, images)
    pre_abs_data, _ = Utilities.process_data_input(pre_abs_data, None)
    rel_data, _ = Utilities.process_data_input(rel_data, None)
    vel_data, _ = Utilities.process_data_input(vel_data, None)

    #Get HCF features of whole dataset
    hcf_data = create_hcf_dataset(pre_abs_data, abs_data, rel_data, vel_data, images, None)


    #Split data by classes
    normal, limp, stagger = Utilities.split_by_class_and_instance(hcf_data)

    #From these datasets, create an average, this will be the actual average
    #right now, to be replaced with an autoencoder function later and compared.
    hcf_normal_ground = Utilities.create_average_data_sample(normal)
    hcf_limp_ground = Utilities.create_average_data_sample(limp)
    hcf_stag_ground = Utilities.create_average_data_sample(stagger)

    #Taking the calculated average, subtract this from every original value to 
    # produce an array of hand crafted features detailing the distance of each 
    # value from it's nearest class. This could be done by averaging the 
    # datasets, or passing the datasets through an autoencoder to create a 
    #representation that's a bit more meaningful.
    ground_truth_dataset = []
    #Enumerate standard HCF data
    for i, row in enumerate(hcf_data):
        ground_truth_row  = row[0:meta + 1]
        for j, value in enumerate(row):
            if j > meta:
                ground_truth_row.append([value - hcf_normal_ground[j],
                                         value - hcf_limp_ground[j],
                                         value - hcf_stag_ground[j]])
        
        ground_truth_dataset.append(ground_truth_row)
    
    Utilities.save_dataset(ground_truth_dataset, joints_output, Utilities.hcf_colnames)
    print("Ground truth dataset created")
    return ground_truth_dataset

def assign_class_labels(num_switches, num_classes, joint_file, joint_output):
    joint_data = Utilities.load(joint_file)
    joint_data = Utilities.apply_class_labels(num_switches, num_classes, joint_data)
    Utilities.save_dataset(joint_data, joint_output)
    return joint_data

def process_empty_frames(joint_file, image_file, joint_output, image_output):
    print("\nProcessing Empty frames...")
    joint_data, image_data = Utilities.process_data_input(joint_file, image_file, cols=Utilities.colnames)
    print(len(joint_data), len(image_data))

    joint_data, image_data = Data_Correction.remove_empty_frames(joint_data, image_data)
    Utilities.save_dataset(joint_data, joint_output)
    Utilities.save_images(joint_data, image_data, image_output)
    print("Empty frame processing complete.")

    return joint_data, image_data

def process_trimmed_frames(joint_file, image_file, joint_output, image_output, trim):
    print("\nProcessing trimmed frames...")
    joint_data, image_data = Utilities.process_data_input(joint_file, image_file)
    joint_data, image_data = Data_Correction.trim_frames(joint_data, image_data, trim = trim)
    Utilities.save_dataset(joint_data, joint_output)
    Utilities.save_images(joint_data, image_data, image_output)
    print("Trimmed frame processing complete.")
    return joint_data, image_data

    
def create_relative_dataset(abs_data, image_data, joint_output, meta = 5):
    print("\nCreating relative value dataset...")
    abs_data, image_data = Utilities.process_data_input(abs_data, image_data)
    rel_data = []
    for i, joints in enumerate(tqdm(abs_data)):
        rel_row = []
        #print("before")
        #Render.render_joints(image_data[i], joints, True)
        for j, coord in enumerate(joints):
            #Ignore metadata
            origin = joints[meta + 1]
            if j <= meta:
                rel_row.append(coord)
            elif j == meta + 1:
                #Origin point, set to 0
                rel_row.append([0,0,0])
            else:
                #Regular coord, append relative to origin
                rel_row.append([coord[0] - origin[0],
                                coord[1] - origin[1],
                                coord[2] - origin[2]])

        rel_data.append(rel_row)
        #print("after")
        #Render.render_joints(image_data[i], rel_row, True)
        #Render.plot3D_joints(rel_row, x_rot=-90, y_rot=180)

    Utilities.save_dataset(rel_data, joint_output)
    print("relative dataset completed.")
    return rel_data

def create_normalized_dataset(joint_data, image_data, joint_output):
    print("\nNormalizing joint data...")
    joint_data, image_data = Utilities.process_data_input(joint_data, image_data, cols=Utilities.colnames)
    #Normalize depth extremities
    joint_data = Data_Correction.normalize_outlier_depths(joint_data, image_data)
    #Normalize outlier values
    joint_data = Data_Correction.normalize_outlier_values(joint_data, image_data, 100)
    Utilities.save_dataset(joint_data, joint_output)
    print("Data normalization complete.")
    return joint_data

def create_scaled_dataset(joint_data, image_data, joint_output):
    #Standardize the scale of the human in the data
    print("\nScaling data...")
    joint_data, image_data = Utilities.process_data_input(joint_data, image_data)
    joint_data = Data_Correction.normalize_joint_scales(joint_data, image_data)
    Utilities.save_dataset(joint_data, joint_output)
    print("Data scale processing complete.")
    return joint_data

def create_flipped_joint_dataset(rel_data, abs_data, images, joint_output, meta = 5):
    print("\nCreating flipped dataset...")
    abs_data, images = Utilities.process_data_input(abs_data, images)
    rel_data, _ = Utilities.process_data_input(rel_data, images)

    sequence_data = Utilities.convert_to_sequences(abs_data)
    rel_sequence_data = Utilities.generate_relative_sequence_data(sequence_data, rel_data)

    flipped_data = [] 
    print("size before flip: ", len(rel_sequence_data), len(sequence_data))
    original_len = len(sequence_data)
    flipped_sequences = []
    for i, seq in enumerate(tqdm(rel_sequence_data)):


        #Get first and last head positions (absolute values)
        first = sequence_data[i][1]
        last = sequence_data[i][-1]

        #This is going from right to left: the ones we want to flip
        #if first[meta+1][1] > last[meta+1][1]:

        #First append regular data:
        for joints in seq:
            flipped_data.append(joints)


        for joints in seq:
            #Append original data
            #Append with metadata
            flipped_joints = joints[0:meta + 1]
            print("sequence: ", flipped_joints[0], original_len, flipped_joints[0] + original_len)
            flipped_joints[0] = flipped_joints[0] + original_len
            for j, joint in enumerate(joints):
                #Flip X value on each individual co-ordinate
                if j > meta:
                    flipped_joints.append([joint[0], -joint[1], joint[2]])
            #Append flipped joints instance to the list
            flipped_sequences.append(flipped_joints)

    for frame in flipped_sequences:
        flipped_data.append(frame)
        #else:
        #Just add the data sequentially for the joints already going from left to right.
        #    for joints in seq:
        #        flipped_data.append(joints)

    
    #Illustrate results
    print("illustrating results: ")
    print("size after flip: ", len(flipped_data))
    for k, joints in enumerate(flipped_data):
        if k > 20 and k < 50:
            pass
            #Render.render_joints(images[k], flipped_data[k], delay=True)
            #Render.render_velocities(abs_data[k], flipped_data[k], images[k])
            #Render.plot3D_joints(flipped_data[k], x_rot=90, y_rot=0)
        elif k > 50:
            break

    Utilities.save_dataset(flipped_data, joint_output)
    print("Data flipping complete.")
    return flipped_data

def create_velocity_dataset(joint_data, image_data, joint_output):
    print("\nCreating velocities dataset...")
    joint_data, image_data = Utilities.process_data_input(joint_data, image_data)
    velocity_dataset = []
    for i, joints in enumerate(tqdm(joint_data)):
        if i+1 < len(joint_data) and i > 0:
            velocity_dataset.append(Utilities.plot_velocity_vectors(None, joint_data[i - 1], joints, joint_data[i + 1]))
        elif i+1 < len(joint_data) and i <= 0:
            velocity_dataset.append(Utilities.plot_velocity_vectors(None, [0], joints, joint_data[i + 1]))
        elif i+1 >= len(joint_data) and i > 0:
            velocity_dataset.append(Utilities.plot_velocity_vectors(None, joint_data[i - 1], joints, [0]))
    
    Utilities.save_dataset(velocity_dataset, joint_output)
    print("Velocity dataset completed.")
    return velocity_dataset

def append_midhip(abs_data, images, joint_output):
    abs_data, images = Utilities.process_data_input(abs_data, images, cols=Utilities.colnames)
    midhip_dataset = []
    for i, joints in enumerate(abs_data):
        midhip_row = list(joints)
        midhip_row.append(Utilities.midpoint(joints[14], joints[15]))
        midhip_dataset.append(midhip_row)
    
        #Display results
        #Render.render_joints(images[i], midhip_row, delay=True)
    
    Utilities.save_dataset(midhip_dataset, joint_output, colnames=Utilities.colnames_midhip)
    return midhip_dataset

def create_bone_dataset(abs_data, images, joint_output, meta = 6):
    abs_data, images = Utilities.process_data_input(abs_data, images)

    bone_dataset = []
    for i, joints in enumerate(abs_data):
        bone_row = list(joints[0:meta])
        for j, coords in enumerate(joints):
            for bone_pair in Utilities.bone_connections:
                if bone_pair[0] + meta == j:
                    #Check if bone is an extremity
                    if bone_pair[1] != -1:
                        #Get direction
                        tmp_vector = [joints[bone_pair[1] + meta][0] - coords[0],
                                    joints[bone_pair[1] + meta][1] - coords[1],
                                    joints[bone_pair[1] + meta][2] - coords[2]]
                        
                        #Then normalize
                        norm = math.sqrt(tmp_vector[0] ** 2 + tmp_vector[1] ** 2 + tmp_vector[2] ** 2)
                        if norm == 0:
                            norm = 1
                        tmp_vector = [tmp_vector[0] / norm, tmp_vector[1] / norm, tmp_vector[2] / norm]
                    else:
                        tmp_vector = [0,0,0]
                    bone_row.append(tmp_vector)
        bone_dataset.append(bone_row)

        #Check solution
        #Render.render_velocities(abs_data[i], bone_row, images[i])

    Utilities.save_dataset(bone_dataset, joint_output, colnames=Utilities.colnames_midhip)
    return bone_dataset                  

def create_joint_angle_dataset(abs_data, images, joint_output, meta = 6):
    print("Creating joint angle dataset (integrated)...")
    joint_angle_dataset = []
    for i, joints in enumerate(tqdm(abs_data)):
        #Add metadata to list and ignore
        joint_angles = joints[0:meta]
        for j, coords in enumerate(joints):
            #This should end up length 2 every time.
            if j > 2:
                coord_angles = None
                connected_joints = []
                for joint in Render.joint_connections:
                    #Find the connections for every joint and build a dataset on that. different joints have different numbers of connections:
                    #minimum 1 (the 6 extremities) and maximum 6 (origin). Strategy is find the first connection of each, then for those with only
                    #one connection, replace the connection with an angle between itself and the origin. (In future, maybe fill third slot with distance
                    # from origin?)
                    #if len(connected_joints) == 2:
                    #    break
                    #Found a connection to this joint
                    if joint[0] == j - 3:
                        connected_joints.append(joints[joint[1]+3])
                    elif joint[1] == j - 3:
                        connected_joints.append(joints[joint[0]+3])

                #This will only apply for hands and feet which have 1 joint connection
                if len(connected_joints) < 2:
                    #Append with the angle between the head and the joint, and the joint and it's connection. This
                    #Should give rough relational angle between the extremities and the origin.

                    #Append a slight offset incase the any of the co-ordinates are identical
                    if coords == connected_joints[0]:
                        coords[0] += 1
                        coords[1] += 1

                    coord_angles = Utilities.ang([coords, joints[3]], [coords, connected_joints[0]])

                    #Render angle
                    angle_plot, angle = Render.get_angle_plot([coords, joints[3]], [coords, connected_joints[0]], 1)
                #And this will only apply for the head joint with 6 connections
                elif len(connected_joints) > 2:
                    #Append with the angle between the head and the mid-rif, connected by the right hip
                    coord_angles = Utilities.ang([coords, joints[15]], [coords, Utilities.midpoint(joints[15], joints[14])])
                    pass
                #This is normal case for the majority of the joints, get the angle between it's two connections.
                else:
                    #Append a slight offset incase the any of the co-ordinates are identical
                    if coords == connected_joints[0] or coords == connected_joints[1]:
                        coords[0] += 1
                        coords[1] += 1

                    coord_angles = Utilities.ang([coords, connected_joints[0]], [coords, connected_joints[1]])
                joint_angles.append(coord_angles)
        joint_angle_dataset.append(joint_angles)
    Utilities.save_dataset(joint_angle_dataset, joint_output)
    print("Joint angle dataset (integrated) Completed.")
    return joint_angle_dataset


def create_decimated_dataset(abs_data, joint_output, images, meta = 10):
    #This will split the data into 2 datasets, top and bottom.
    abs_data, images = Utilities.process_data_input(abs_data, images)
    dataset = []

    for i, joints in enumerate(tqdm(abs_data)):
        top_row = list(joints[0:meta])
        #Append the mid-hip to bottom row in place of the origin ::::  This is now done earlier
        #bottom_row.append(Utilities.midpoint(joints[14], joints[15]))

        for j, coords in enumerate(joints):
            if j > 10:
                top_row.append(coords)

        #Render.render_joints(images[i], top_row, delay=True)
        dataset.append(top_row)

    #Extract correct column names
    top_colnames = list(Utilities.colnames_midhip[0:meta])
    top_colnames += Utilities.colnames_midhip[11:]

    Utilities.save_dataset(dataset, joint_output + "_decimated", top_colnames)

    print("Decimated dataset completed.")
    return dataset

def create_2_regions_dataset(abs_data, joint_output, images, meta = 6, size = 9):
    #This will split the data into 2 datasets, top and bottom.
    abs_data, images = Utilities.process_data_input(abs_data, images)
    top_dataset = []
    bottom_dataset = []

    for i, joints in enumerate(tqdm(abs_data)):
        top_row = list(joints[0:meta])
        bottom_row = list(joints[0:meta])
        empties = list(np.zeros(size))
        #Append the mid-hip to bottom row in place of the origin ::::  This is now done earlier
        #bottom_row.append(Utilities.midpoint(joints[14], joints[15]))

        for j, coords in enumerate(joints):
            if j >= 17:
                bottom_row.append(coords)
                top_row.append(empties)
            elif j > 5:
                top_row.append(coords)
                bottom_row.append(empties)

        #Render.render_joints(images[i], top_row, delay=True)
        top_dataset.append(top_row)
        bottom_dataset.append(bottom_row)

    #Extract correct column names
    top_colnames = list(Utilities.colnames[0:meta])
    bottom_colnames = list(Utilities.colnames[0:meta])
    bottom_colnames += ["joint_0"]

    top_colnames += Utilities.colnames[5: 16]
    bottom_colnames += Utilities.colnames[17:]

    Utilities.save_dataset(top_dataset, joint_output + "_top", top_colnames)
    Utilities.save_dataset(bottom_dataset, joint_output + "_bottom", bottom_colnames)
    print("Regions dataset (top and bottom) completed.")
    return top_dataset, bottom_dataset

def append_specific_joints(my_list, joints, indices, size = 9):
    empties = list(np.zeros(size))
    for i in range(len(joints)):
        if i in indices:
            my_list.append(joints[i])
        else:
            if i > 5:
                my_list.append(empties)
    return my_list

def create_5_regions_dataset(abs_data, joint_output, images, meta = 5, size = 9):
    abs_data, images = Utilities.process_data_input(abs_data, images)
    #The regions are left_arm, left_leg, right_arm, right_leg, head, so essentially exodia.
    region_datasets = [[],[],[],[],[]]

    for i, joints in enumerate(tqdm(abs_data)):
        region_rows = [[],[],[],[],[]]
        
        #Append metadata to each region
        for j, region in enumerate(region_rows):
            #print("region rows: ", region_rows[j], list(joints[0:3]))
            region_rows[j] = list(joints[0:meta + 1])
        
        #region_rows[0] += joints[3:8] # Head joints
        for index, k in enumerate(joints):
            if index > meta and index < 11:
                region_rows[0].append(k)
            else:
                if index > meta:
                    region_rows[0].append(list(np.zeros(size)))
        #region_rows[0] = [k for index, k in enumerate(joints) if index > meta and index < 11 else [0,0,0]]
                                                                        #8 is 11
        region_rows[1] = append_specific_joints(region_rows[1], joints, [11,13,15], size=size)
        region_rows[2] = append_specific_joints(region_rows[2], joints, [12,14,16], size=size)
        region_rows[3] = append_specific_joints(region_rows[3], joints, [17,19,21], size=size)
        region_rows[4] = append_specific_joints(region_rows[4], joints, [18,20,22], size=size)

        #Check I've got the right coordinates
        #for j, region in enumerate(region_rows):
        #    Render.render_joints(images[i], region_rows[j], delay=True)
        
        for k, r in enumerate(region_datasets):
            r.append(region_rows[k])
    
    output_suffixes = ["head", "r_arm", "l_arm", "r_leg", "l_leg"]
    #col_names = ["Instance", "No_In_Sequence", "Class", 'Freeze', 'Obstacle', 'Person', "Joint_0", "Joint_1", "Joint_2"]
    #head_col_names = ["Instance", "No_In_Sequence", "Class",'Freeze', 'Obstacle', 'Person',  "Joint_0", "Joint_1", "Joint_2", "Joint_3", "Joint_4"]

    for i, r in enumerate(region_datasets):
        #if i == 0:
        #    output_cols = head_col_names
        #else:
        #    output_cols = col_names

        #print("lens: ", len(output_cols), len(r), len(r[0]))
        Utilities.save_dataset(r, joint_output + output_suffixes[i])
        
    print("Regions dataset (5-tier) completed.")
    return region_datasets
                

def transform_gait_cycle_data(cycles, data):
    gait_cycles = []
    joint_counter = 0
    for cycle in cycles:
        new_cycle = []
        for frame in cycle:
            new_cycle.append(data[joint_counter])
            joint_counter += 1
        gait_cycles.append(copy.deepcopy(new_cycle))
    
    return gait_cycles

def set_gait_cycles(data, preset_cycle):
    new_cycles = []
    data_iter = 0
    print("len data in set should be 1960: ", len(data))
    for i, cycle in enumerate(preset_cycle):
        new_cycle = []
        for j, frame in enumerate(cycle):
            #print("data iter: ", data_iter, len(preset_cycle), len(cycle))
            new_cycle.append(data[data_iter])
            data_iter += 1
        new_cycles.append(new_cycle)
    
    print("final lens: ", len(new_cycles), len(preset_cycle))
    return new_cycles
    
def create_hcf_dataset(pre_abs_joints, abs_joints, rel_joints, abs_veljoints, images, joints_output, meta = 5):
    abs_joint_data, images = Utilities.process_data_input(abs_joints, images)
    pre_scale, _ = Utilities.process_data_input(pre_abs_joints, None)
    rel_joint_data, _ =  Utilities.process_data_input(rel_joints, None)
    abs_veljoint_data, _ =  Utilities.process_data_input(abs_veljoints, None)

    print("Building HCF Dataset...", len(abs_joint_data), len(rel_joint_data), len(pre_abs_joints))
    pre_gait_cycles = hcf.get_gait_cycles(pre_abs_joints, None)
    gait_cycles = set_gait_cycles(abs_joint_data, pre_gait_cycles)
    rel_gait_cycles = set_gait_cycles(rel_joint_data, pre_gait_cycles)

    print("gait cycle lens: ", len(pre_gait_cycles), len(gait_cycles), len(rel_gait_cycles))
    #trend = hcf.get_knee_chart_polynomial(knee_data_cycles)
    knee_data_cycles = Utilities.build_knee_joint_data(pre_gait_cycles, images)
    knee_data_coeffs = Render.chart_knee_data(knee_data_cycles, False)


    #print("number of total gait cycles: ", len(gait_cycles))
    #Then for every gait cycle create a new instance with the following features: 
    #Cadences returns a scalar for every gait cycle returning the number of steps 
    #cadences = get_cadence(gait_cycles, images)

    #Height of feet above the ground
    #returns feet heights using distance between each foot joint and head in relative
    #terms using absolute data. returned as array of 2 element lists for each foot
    feet_heights = hcf.get_feet_height(gait_cycles, images)


    #Time leg off of ground + time both feet not moving
    #if foot velocity > 0 + threshold and math.dist(foot, head) > threshold then foot is in motion, add to vector
    #once this no longer holds, change colour and show
    #implement for left and right leg
    times_LOG, times_not_moving = hcf.get_time_LofG(gait_cycles, abs_veljoint_data, images)

    #Speed
    #get absolute values of the head at first frame vs last, divide number of frames by that distance
    speeds = hcf.get_speed(gait_cycles, images)

    #Stride length + variance
    #get max and min from left leg relative to head
    #get corresponding values in absolute values for leg
    #math.dist to get the distance between these two values
    #repeat for both legs
    stride_lengths, stride_ratios = hcf.get_stride_lengths(rel_gait_cycles, images, gait_cycles)
    stride_gaps, max_gaps = hcf.get_stride_gap(gait_cycles, images)
    #Combine all hand crafted features into one concrete dataset, save and return it.
    gait_cycles_dataset = []
    for i, cycle in enumerate(gait_cycles):
        #Add metadata
        hcf_cycle = cycle[0][0:meta + 1]#[0], cycle[0][1], cycle[0][2]]
        #hcf_cycle.append(cadences[i])
        hcf_cycle.append(feet_heights[i][0])
        hcf_cycle.append(feet_heights[i][1])
        hcf_cycle.append(times_LOG[i][0])
        hcf_cycle.append(times_LOG[i][1])
        hcf_cycle.append(times_not_moving[i])
        hcf_cycle.append(speeds[i])
        hcf_cycle.append(stride_gaps[i])
        #hcf_cycle.append(max_gaps[i])
        #hcf_cycle.append(stride_lengths[i][0])
        hcf_cycle.append(stride_lengths[i][1])
        hcf_cycle.append(max_gaps[i])
        for c in knee_data_coeffs[i]:
            hcf_cycle.append(c)
        #hcf_cycle.append(stride_ratios[i])
        gait_cycles_dataset.append(copy.deepcopy(hcf_cycle))

   # hcf_names = ["Instance", "No_In_Sequence", "Class", "Feet_Height_0", "Feet_Height_1",
   #              "Time_LOG_0", "Time_LOG_1", "Time_No_Movement", "Speed", "Stride_Gap", "Stride_Length", "Max_Gap", 'l_co 1',
   #              'l_co 2', 'l_co 3', 'l_co 4', 'l_co 5', 'l_co 6', 'l_co 7', 'r_co 1', 'r_co 2', 'r_co 3', 'r_co 4', 'r_co 5', 'r_co 6', 'r_co 7']
    if joints_output != None:
        print("len hcf: ", len(gait_cycles_dataset))
        Utilities.save_dataset(gait_cycles_dataset, joints_output, Utilities.hcf_colnames)
    print("HCF dataset completed.")
    return gait_cycles_dataset


def create_dummy_dataset(data, output_name):
    #Get single datapoint of all 3 classes
    data, _ = Utilities.process_data_input(data, None)

    #Apply gaussian noise to each 1000*
    mean = 0  # Mean of the Gaussian distribution
    std_dev = 1 

    sequences = []
    current_sequence = 0
    sequence = []
    for i, example in enumerate(data):
        if example[0] == current_sequence:
            sequence.append(example)
        else:
            sequences.append(copy.deepcopy(sequence))
            sequence = []
            sequence.append(example)
            current_sequence += 1
    sequences.append(sequence)

    print("should be 59: ", len(sequences))
    original_len = len(sequences)
    noise_sequences = []
    novel_sequences = []
    for i, sequence in enumerate(sequences):

        #noise_sequences.append(sequence)
        #First: Add the original frames
        for frame in sequence:
            noise_sequences.append(frame)


        for j in range(3):
            for frame in sequence:
                frame_metadata = frame[0:6]
                frame_metadata[0] = frame_metadata[0] + original_len
                #frame_metadata[0] = instance_counter
                joints_frame = frame[6:]
                noisy_frame = joints_frame + np.random.normal(mean, std_dev, (len(joints_frame), len(joints_frame[0])))
                
                #Convert from numpy arrays to lists so it saves to csv nicely
                noisy_frame = list(noisy_frame)
                for k, tmp in enumerate(noisy_frame):
                    noisy_frame[k] = list(noisy_frame[k])

                #Unravel the denoised frame and attach to the metadata
                for f in noisy_frame:
                    frame_metadata.append(f)

                #done = 5/0
                novel_sequences.append(frame_metadata)
        
    for frame in novel_sequences:
        noise_sequences.append(frame)
                #instance_counter += 1
        #noise_sequences.append(noisy_sequence)

    
    print("final number of fake examples, should be 959 * 3: ", len(noise_sequences))
    #Save as csv
    Utilities.save_dataset(noise_sequences, output_name)
    return noise_sequences


def interpolate_gait_cycle(data_cycles, joint_output, step = 5):
    inter_cycles = []
    for a, cycle in enumerate(data_cycles):
        inter_cycle = []
        #print("original cycle length: ", len(cycle))
        for i, frame in enumerate(cycle):
            #Add the frame first
            inter_cycle.append(frame)

            #Ignore the last frame for interpolation
            if i < len(cycle) - 1:
                inter_frames = interpolate_coords(frame, cycle[i + 1], step)
                #Unwrap and add to full cycle 
                for j in range(step):
                    inter_cycle.append(inter_frames[j])
                    #print("inter frames: ", inter_frames[j])
                    #print("normal frame: ", frame)
                    #print("types: ", type(inter_frames[0][6]), type(frame[6]))
                    #print("actual: ", inter_frames[0][6], frame[6])
                    #stop = 5/0

        #print("new cycle should be ", step, " times longer: ", len(inter_cycle))
           # if i > 1:
           #     for c in inter_cycle:
           #         print("here: ", type(c), c)
                #stop = 5/0
        inter_cycles.append(inter_cycle)
    
    print("cycle length should be same: ", len(inter_cycles), len(data_cycles))
    return inter_cycles


def interpolate_coords(start_frame, end_frame, step):
    #start_frame = np.array(start_frame)
    #end_frame = np.array(end_frame)

    # Calculate the step size for interpolation
    inter_frames = []
    for i in range(1, step + 1):
        inter_frame = copy.deepcopy(start_frame)
        for j, coord in enumerate(start_frame):
            if j > 5:
                step_size = (np.array(end_frame[j]) - np.array(coord)) / (step + 1)
                # Perform interpolation and create the new lists
                interpolated_coord = coord + i * step_size
                #print("interpolated coord 1: ", type(interpolated_coord), interpolated_coord)   
                listed = list(interpolated_coord)
                #print("\ninterpolated coord 2: ", type(listed), listed)
                inter_frame[j] = listed
        inter_frames.append(inter_frame)


        
    return inter_frames

            