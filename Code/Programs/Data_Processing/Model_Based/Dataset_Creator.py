import Programs.Data_Processing.Model_Based.Data_Correction as Data_Correction
import Programs.Data_Processing.Model_Based.Utilities as Utilities
import Programs.Data_Processing.Model_Based.Render as Render
import Programs.Data_Processing.Model_Based.HCF as hcf
from tqdm import tqdm
import math
import copy 

def combine_datasets(rel_data, vel_data, angle_data, images, joints_output):
    print("Combining datasets...")
    rel_data, images = Utilities.process_data_input(rel_data, images)
    vel_data, _ = Utilities.process_data_input(vel_data, None)
    angle_data, _ = Utilities.process_data_input(angle_data, None)

    combined_dataset = []
    for i, row in enumerate(tqdm(rel_data)):
        #Metadata is the same as usual
        combined_row = row[0:3]
        for j, joint in enumerate(row):
            if j > 2:
                #print("row before: ", combined_row[0:5])
                combined_row.append([joint, vel_data[i][j], [0,0,angle_data[i][j]]])
                #print("row after: ", combined_row[0:5])
        
        #print("final combined row: ", combined_row[0:5])
        combined_dataset.append(combined_row)
    
    print("Completing combined dataset.")
    Utilities.save_dataset(combined_dataset, joints_output)
    return combined_dataset


def create_ground_truth_dataset(pre_abs_data, abs_data, rel_data, vel_data, hcf_data, images, joints_output):
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
        ground_truth_row  = row[0:3]
        for j, value in enumerate(row):
            if j > 2:
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
    joint_data, image_data = Utilities.process_data_input(joint_file, image_file)
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

    
def create_relative_dataset(abs_data, image_data, joint_output):
    print("\nCreating relative value dataset...")
    abs_data, image_data = Utilities.process_data_input(abs_data, image_data)
    rel_data = []
    for i, joints in enumerate(tqdm(abs_data)):
        rel_row = []
        #print("before")
        #Render.render_joints(image_data[i], joints, True)
        for j, coord in enumerate(joints):
            #Ignore metadata
            origin = joints[3]
            if j < 3:
                rel_row.append(coord)
            elif j == 3:
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
    joint_data, image_data = Utilities.process_data_input(joint_data, image_data)
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

def create_flipped_joint_dataset(rel_data, abs_data, images, joint_output):
    print("\nCreating flipped dataset...")
    abs_data, images = Utilities.process_data_input(abs_data, images)
    rel_data, _ = Utilities.process_data_input(rel_data, images)

    sequence_data = Utilities.convert_to_sequences(abs_data)
    rel_sequence_data = Utilities.generate_relative_sequence_data(sequence_data, rel_data)

    flipped_data = [] 
    for i, seq in enumerate(tqdm(rel_sequence_data)):

        #Get first and last head positions (absolute values)
        first = sequence_data[i][1]
        last = sequence_data[i][-1]

        #This is going from right to left: the ones we want to flip
        if first[3][1] > last[3][1]:
            for joints in seq:
                #Append with metadata
                flipped_joints = joints[0:3]
                for j, joint in enumerate(joints):
                    #Flip X value on each individual co-ordinate
                    if j > 2:
                        flipped_joints.append([joint[0], -joint[1], joint[2]])
                #Append flipped joints instance to the list
                flipped_data.append(flipped_joints)

        else:
        #Just add the data sequentially for the joints already going from left to right.
            for joints in seq:
                flipped_data.append(joints)

    
    #Illustrate results
    print("illustrating results: ")
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
            velocity_dataset.append(Utilities.plot_velocity_vectors(image_data[i], joint_data[i - 1], joints, joint_data[i + 1]))
        elif i+1 < len(joint_data) and i <= 0:
            velocity_dataset.append(Utilities.plot_velocity_vectors(image_data[i], [0], joints, joint_data[i + 1]))
        elif i+1 >= len(joint_data) and i > 0:
            velocity_dataset.append(Utilities.plot_velocity_vectors(image_data[i], joint_data[i - 1], joints, [0]))
    
    Utilities.save_dataset(velocity_dataset, joint_output)
    print("Velocity dataset completed.")
    return velocity_dataset


def create_joint_angle_dataset(abs_data, images, joint_output):
    print("Creating joint angle dataset (integrated)...")
    joint_angle_dataset = []
    for i, joints in enumerate(tqdm(abs_data)):
        #Add metadata to list and ignore
        joint_angles = joints[0:3]
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

def create_2_regions_dataset(abs_data, joint_output, images):
    #This will split the data into 2 datasets, top and bottom.
    abs_data, images = Utilities.process_data_input(abs_data, images)
    top_dataset = []
    bottom_dataset = []

    for i, joints in enumerate(tqdm(abs_data)):
        top_row = list(joints[0:3])
        bottom_row = list(joints[0:3])
        #Append the mid-hip to bottom row in place of the origin
        bottom_row.append(Utilities.midpoint(joints[14], joints[15]))

        for j, coords in enumerate(joints):
            if j >= 14:
                bottom_row.append(coords)
            elif j > 2:
                top_row.append(coords)

        #Render.render_joints(images[i], top_row, delay=True)
        top_dataset.append(top_row)
        bottom_dataset.append(bottom_row)

    #Extract correct column names
    top_colnames = list(Utilities.colnames[0:3])
    bottom_colnames = list(Utilities.colnames[0:3])
    bottom_colnames += ["joint_0"]

    top_colnames += Utilities.colnames[2: 13]
    bottom_colnames += Utilities.colnames[14:]

    Utilities.save_dataset(top_dataset, joint_output + "_top.csv", top_colnames)
    Utilities.save_dataset(bottom_dataset, joint_output + "_bottom.csv", bottom_colnames)
    print("Regions dataset (top and bottom) completed.")
    return top_dataset, bottom_dataset

def append_specific_joints(my_list, joints, indices):
    for i in indices:
        my_list.append(joints[i])
    return my_list

def create_5_regions_dataset(abs_data, joint_output, images):
    abs_data, images = Utilities.process_data_input(abs_data, images)
    #The regions are left_arm, left_leg, right_arm, right_leg, head, so essentially exodia.
    region_datasets = [[],[],[],[],[]]

    for i, joints in enumerate(tqdm(abs_data)):
        region_rows = [[],[],[],[],[]]
        
        #Append metadata to each region
        for j, region in enumerate(region_rows):
            #print("region rows: ", region_rows[j], list(joints[0:3]))
            region_rows[j] = list(joints[0:3])
        
        #region_rows[0] += joints[3:8] # Head joints
        region_rows[0] += [k for index, k in enumerate(joints) if index > 2 and index < 8]

        region_rows[1] = append_specific_joints(region_rows[1], joints, [8,10,12])
        region_rows[2] = append_specific_joints(region_rows[2], joints, [9,11,13])
        region_rows[3] = append_specific_joints(region_rows[3], joints, [14,16,18])
        region_rows[4] = append_specific_joints(region_rows[4], joints, [15,17,19])

        #Check I've got the right coordinates
        #for j, region in enumerate(region_rows):
        #    Render.render_joints(images[i], region_rows[j], delay=True)
        
        for k, r in enumerate(region_datasets):
            r.append(region_rows[k])
    
    output_suffixes = ["head", "r_arm", "l_arm", "r_leg", "l_leg"]
    col_names = ["Instance", "No_In_Sequence", "Class", "Joint_0", "Joint_1", "Joint_2"]
    head_col_names = ["Instance", "No_In_Sequence", "Class", "Joint_0", "Joint_1", "Joint_2", "Joint_3", "Joint_4"]

    for i, r in enumerate(region_datasets):
        if i == 0:
            output_cols = head_col_names
        else:
            output_cols = col_names

        Utilities.save_dataset(r, joint_output + output_suffixes[i] + ".csv", output_cols)
        
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

def create_hcf_dataset(pre_abs_joints, abs_joints, rel_joints, abs_veljoints, images, joints_output, check = 0):
    abs_joint_data, images = Utilities.process_data_input(abs_joints, images)
    pre_scale, _ = Utilities.process_data_input(pre_abs_joints, None)
    rel_joint_data, _ =  Utilities.process_data_input(rel_joints, None)
    abs_veljoint_data, _ =  Utilities.process_data_input(abs_veljoints, None)

    print("Building HCF Dataset...")
    #Gait cycle has instances of rows: each instance contains an array of rows
    #denoting all of the frames in their respective gait cycle, appended with their
    #metadata at the start
    #if check == 1:
    #    print(pre_scale[0])

    pre_gait_cycles = hcf.get_gait_cycles(pre_scale, images)

    #Experiment with getting and then charting knee data
    #for i in range(len(gait_cycles)):
    #    if i <= 2:
    #        angles = Utilities.build_knee_joint_data(gait_cycles[i])
    #        Utilities.chart_knee_data(angles)

    #Replicate this pattern for the relative gait cycle and scaled gait data (if youn run it directly, 
    #relative gait will return very marginally different gait cycles.)
    gait_cycles = transform_gait_cycle_data(pre_gait_cycles, abs_joint_data)
    rel_gait_cycles = transform_gait_cycle_data(pre_gait_cycles, rel_joint_data)

    #rel_gait_cycles = []
    #joint_counter = 0
    #for cycle in gait_cycles:
    #    new_cycle = []
    #    for frame in cycle:
    #        new_cycle.append(rel_joint_data[joint_counter])
    #        joint_counter += 1
    #    rel_gait_cycles.append(copy.deepcopy(new_cycle))


    print("number of total gait cycles: ", len(gait_cycles))
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
        hcf_cycle = cycle[0][0:3]#[0], cycle[0][1], cycle[0][2]]
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
        #hcf_cycle.append(stride_ratios[i])
        gait_cycles_dataset.append(copy.deepcopy(hcf_cycle))

    col_names = ["Instance", "No_In_Sequence", "Class", "Feet_Height_0", "Feet_Height_1",
                 "Time_LOG_0", "Time_LOG_1", "Time_No_Movement", "Speed", "Stride_Gap", "Stride_Length", "Max_Gap"]
    if joints_output != None:
        Utilities.save_dataset(gait_cycles_dataset, joints_output, col_names)
    print("HCF dataset completed.")
    return gait_cycles_dataset

