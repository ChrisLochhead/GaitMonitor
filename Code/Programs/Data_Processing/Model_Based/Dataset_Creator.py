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
                    combined_row.append([joint[0], joint[1], joint[2],
                                        vel_data[i][j][0], vel_data[i][j][1], vel_data[i][j][2], 
                                        angle_data[i][j][0], angle_data[i][j][1], angle_data[i][j][2] ])

        combined_dataset.append(combined_row)
    
    print("Completing combined dataset.")
    Utilities.save_dataset(combined_dataset, joints_output)
    return combined_dataset

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
    #Smooth values that are visually a bit haywire on the arms and legs
    joint_data = Data_Correction.smooth_unlikely_values(joint_data)
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

def create_flipped_joint_dataset(rel_data, abs_data, images, joint_output, meta = 5, double_size = True, already_sequences = False):
    print("\nCreating flipped dataset...")
    abs_data, images = Utilities.process_data_input(abs_data, images)
    rel_data, _ = Utilities.process_data_input(rel_data, images)

    if already_sequences == False:
        sequence_data = Utilities.convert_to_sequences(abs_data)
    else:
        sequence_data = abs_data

    print("lens: ", len(sequence_data), len(rel_data))
    totals = [0,0]
    for l in sequence_data:
        totals[0] += len(l)
    for a in rel_data:
        totals[1] += len(a)

    print("totals: ", totals)
    if already_sequences == False:
        rel_sequence_data = Utilities.generate_relative_sequence_data(sequence_data, rel_data)
    else:
        rel_sequence_data = rel_data

    flipped_data = []
    print("size before flip: ", len(rel_sequence_data), len(sequence_data))
    original_len = len(sequence_data)
    flipped_sequences = []
    for i, seq in enumerate(tqdm(rel_sequence_data)):


        #Get first and last head positions (absolute values)
        first = sequence_data[i][1]
        last = sequence_data[i][-1]

        #This is going from right to left: the ones we want to flip
        if (first[meta+1][1] > last[meta+1][1] and double_size == False) or double_size == True:

            #First append regular data:
            if double_size:
                for joints in seq:
                    flipped_data.append(joints)

            for joints in seq:
                flipped_joints = joints[0:meta + 1]
                if double_size:
                    flipped_joints[0] = flipped_joints[0] + original_len
                for j, joint in enumerate(joints):
                    #Flip X value on each individual co-ordinate
                    if j > meta:
                        flipped_joints.append([joint[0], -joint[1], joint[2]])
                #Append flipped joints instance to the list
                flipped_sequences.append(flipped_joints)
        else:
            for joints in seq:
                flipped_sequences.append(joints)
                
    for frame in flipped_sequences:
        flipped_data.append(frame)
    
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

    for i, r in enumerate(region_datasets):
        Utilities.save_dataset(r, joint_output + output_suffixes[i])
        
    print("Regions dataset (5-tier) completed.")
    return region_datasets
                

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
    sequence = []
    for i, example in enumerate(data):
        #Just add the first one as usual
        if i == 0:
            sequence.append(example)
        else:
            #if the current sequence value is lower than the last, then it's a new sequence
            if example[1] < data[i-1][1]:
                sequences.append(copy.deepcopy(sequence))
                sequence = []
                sequence.append(example)
            else:
                sequence.append(example)

    sequences.append(sequence)

    noise_sequences = []
    novel_sequences = []

    print("info: ", len(sequences), len(sequences[0]))

    #Get total number of people
    no_people = sequences[-1][0][5]
    if no_people <= 0:
        no_people =1
    print("should be 16: ", no_people)

    scaling_factors = [0 for i in range(no_people)]
    frame_counts = [0 for i in range(no_people)]
    #Minimum number of examples you want per person in frames
    threshold = 13000

    for sequence in sequences:

        for frame in sequence:
            #Add -1 for weightgait
            #Edge case: producing single person datasets sets person to -1
            if frame[5] < 0:
                frame[5] = 0
            #print("what is this: ", frame[5], len(frame_counts), frame_counts)
            frame_counts[frame[5]-1] += 1

    print("final frame counts per person: ", frame_counts)
    print("total frames: ", sum(frame_counts))

    for i, factor in enumerate(scaling_factors):
        #Calculate what I need to multiply by to reach a minimum of 13000 frames (or 2200ish sequences of 6)
        #5 (chris) should be close to 5, Erin 12 and Elisa 10
        scaling_factors[i] = int(threshold /frame_counts[i])

    print("scaling factors: ", scaling_factors)
    #done = 5/0

    scaling_iter = 0
    print("Num sequences should be 1917", len(sequences))
    original_len = len(sequences)
    for i, sequence in enumerate(sequences):
        #First: Add the original frames
        if i > 0 and (i + 1) % 60 == 0:
            scaling_iter +=1
            if scaling_iter > len(scaling_factors)-1:
                scaling_iter = 0
            print("changing scale iter: ", i, len(sequences), len(scaling_factors))
            print("factor now " ,scaling_factors[scaling_iter])
        if i %100 == 0:
            print("whats i right now: ", i)

        for frame_index, frame in enumerate(sequence):
            noise_sequences.append(frame)
        
        for j in range(scaling_factors[scaling_iter]):
            for frame in sequence:
                frame_metadata = frame[0:6]
                frame_metadata[0] = frame_metadata[0] + original_len + (j + 1)
                joints_frame = frame[6:]
                noisy_frame = joints_frame + np.random.normal(mean, std_dev, (len(joints_frame), len(joints_frame[0])))
                
                #Convert from numpy arrays to lists so it saves to csv nicely
                noisy_frame = list(noisy_frame)
                for k, tmp in enumerate(noisy_frame):
                    noisy_frame[k] = list(noisy_frame[k])

                #Unravel the denoised frame and attach to the metadata
                for f in noisy_frame:
                    frame_metadata.append(f)

                novel_sequences.append(frame_metadata)
    
    print("length aFTER ADDING ALL NORmal: ", len(noise_sequences))
    for frame in novel_sequences:
        noise_sequences.append(frame)

    print("final number of fake examples: ", len(noise_sequences))
    #stop = 5/0
    Utilities.save_dataset(noise_sequences, output_name)
    return noise_sequences


def interpolate_gait_cycle(data_cycles, joint_output, step = 5, restrict_cycle = False):
    inter_cycles = []
    min_cycle_count = min(len(sub_list) for sub_list in data_cycles) - 1

    print("len data cycles: ", len(data_cycles))
    for a, cycle in enumerate(data_cycles):
        
        inter_cycle = []
        #print("original cycle length: ", len(cycle))
        for i, frame in enumerate(cycle):
            if i < min_cycle_count or restrict_cycle == False:
                #Add the frame first
                inter_cycle.append(frame)

                #Ignore the last frame for interpolation
                if i < len(cycle) - 1:
                    inter_frames = interpolate_coords(frame, cycle[i + 1], step)
                    #Unwrap and add to full cycle 
                    for j in range(step):
                        inter_cycle.append(inter_frames[j])

        inter_cycles.append(inter_cycle)
    
    print("cycle length should be same: ", len(inter_cycles), len(data_cycles))
    save_cycles = []
    for c in inter_cycles:
        for f in c:
            save_cycles.append(f)
    if joint_output != None:
        Utilities.save_dataset(save_cycles, joint_output)
    return inter_cycles


def interpolate_coords(start_frame, end_frame, step):
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

def check_within_radius(point1, point2, radius):
    if len(point1) != 3 or len(point2) != 3:
        raise ValueError("Both points should have exactly 3 dimensions.")
    
    #3D is no good for this
    distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)# + (point1[2] - point2[2])**2)
    return distance <= radius

def get_average_sequence(data):
    #Get the first sequence and divide it by the number of sequences
    result = data[0]
    for i , frame in enumerate(result):
        for j, coords in enumerate(frame):
            if j > 5:
                result[i][j] = [val / len(data) for val in coords]

    #Add every subsequent frame weighted by the number of sequences in the data
    for i, sequence in enumerate(data):
        if i > 0:
            for j, frame in enumerate(sequence):
                for k, coords in enumerate(frame):
                    if k > 5:
                        new_addition = [val / len(data) for val in coords]
                        try:
                            result[j][k] = [x + y for x, y in zip(result[j][k], new_addition)]
                        except:
                            pass
    
    return result
                    
def subtract_skeleton(rel_data, joint_output, base_output):

    rel_sequences = Utilities.convert_to_sequences(rel_data)

    rel_sequences = interpolate_gait_cycle(rel_sequences, base_output, 0, restrict_cycle=False) # try just cutting all to minimum size first, then by padding

    overlay_sequences = [s for i, s in enumerate(rel_sequences) if i % 60 == 0]
    for i, sequence in enumerate(rel_sequences):
        if i % 60 == 0:
            overlay_sequences.append(get_average_sequence(rel_sequences[i:i+10]))

    overlay_iter = 0
    sequence_counter = 0
    for i, sequence in enumerate(rel_sequences):
        for j, frame in enumerate(sequence):
            for k, coord in enumerate (frame):
                if k> 5:
                    #Check if coord and overlay[j][k] are within a radius of eachother, ignoring the first 10
                    try:
                        if check_within_radius(coord, overlay_sequences[overlay_iter][j][k], 0):# results were on 50. minor was 15
                            #print("detected within raidus: ", coord, overlay_sequence[j][k])
                            pass
                            #rel_sequences[i][j][k] = [0.0, 0.0, 0.0]
                    except:
                        pass
        if i % 60 == 0 and i != 0:
            overlay_iter += 1
            print("overlay iter: ", overlay_iter)
            sequence_counter = 0
        else:
            sequence_counter += 1
        

    #Unwrap sequences
    final_data = []
    for i, sequence in enumerate(rel_sequences):
        for frame in sequence:
            final_data.append(frame)
            #Render.plot3D_joints(frame, metadata=6)

    Utilities.save_dataset(final_data, joint_output)
    return final_data

def convert_person_to_type(data, joint_output):
    for i, row in enumerate(data):
        #Types:
        #Class 0, freeze 0, obstacle 0 = 0
        #Class 0, freeze 1, obstacle 0 = 1    
        #Class 0, freeze 0, obstacle 1 = 2  
        #Class 1, freeze 0, obstacle 0 = 3
        #Class 1, freeze 1, obstacle 0 = 4
        #Class 1, freeze 0, obstacle 1 = 5  
        #Class 2, freeze 0, obstacle 0 = 6
        #Class 2, freeze 1, obstacle 0 = 7
        #Class 2, freeze 0, obstacle 1 = 8

        if row[2] == 0:
            #Freeze 0
            if row[3] == 0:
                #Obstacle 0
                if row[4] == 0:
                    #Class 0, freeze 0, obstacle 0 = 0
                    data[i][5] = 0
                else:
                    #Class 0, freeze 0, obstacle 0 = 1
                    data[i][5] = 2
            else:
                data[i][5] = 1
        elif row[2] == 1:
            #Freeze 0
            if row[3] == 0:
                #Obstacle 0
                if row[4] == 0:
                    #Class 1, freeze 0, obstacle 0 = 0
                    data[i][5] = 3
                else:
                    #Class 1, freeze 0, obstacle 0 = 1
                    data[i][5] = 5
            else:
                data[i][5] = 4
        elif row[2] == 2:
            #Freeze 0
            if row[3] == 0:
                #Obstacle 0
                if row[4] == 0:
                    #Class 0, freeze 0, obstacle 0 = 0
                    data[i][5] = 6
                else:
                    #Class 0, freeze 0, obstacle 0 = 1
                    data[i][5] = 8
            else:
                data[i][5] = 7

    Utilities.save_dataset(data, joint_output)
    return data


def split_into_streams(data, joint_output_r, joint_output_v, joint_output_b):
    rel_data = []
    vel_data = []
    bone_data = []

    for i, row in enumerate(data):
        rel_row = row[0:6]
        vel_row = row[0:6]
        bone_row = row[0:6]

        for j, coord in enumerate(row):
            if j > 5:
                rel_row.append(coord[0:3])
                vel_row.append(coord[3:6])
                bone_row.append(coord[6:9])
        
        rel_data.append(rel_row)
        vel_data.append(vel_row)
        bone_data.append(bone_row)

    Utilities.save_dataset(rel_data, joint_output_r)
    Utilities.save_dataset(vel_data, joint_output_v)
    Utilities.save_dataset(bone_data, joint_output_b)
    


def assign_person_number(data_to_append, data, joint_output, no, start_instance):
    current_instance = start_instance + 1
    for i, row in enumerate(data):
        data[i][5] = no
        if i > 0:
            if row[1] < data[i-1][1]:
                current_instance +=1
        
        data[i][0] = current_instance

    if data_to_append != None:
        #Append to a master dataset
        for d in data:
            data_to_append.append(d)

    Utilities.save_dataset(data_to_append, joint_output)
    return current_instance, data_to_append 
