'''
This file contains all functions involving the creation of new data-streams for passing into the networks.
'''
#imports
from tqdm import tqdm
import numpy as np
import copy 
#dependencies
import Programs.Data_Processing.Data_Correction as Data_Correction
import Programs.Data_Processing.Utilities as Utilities
import Programs.Data_Processing.Render as Render
import Programs.Data_Processing.HCF as hcf

def avg_coord(joint_data):
    '''
    Simple utility function for combining datasets, returning the average coord from each column

    Arguments
    ---------
    data : List(List())
        Original joints dataset

    Returns
    -------
    List()
        List of the averages of each column
    '''
    total_coord = joint_data[0]
    for i, d in enumerate(joint_data):
        if i > 0:
            for j, coord in enumerate(d):
                total_coord[j] += coord
    for i, coord in enumerate(total_coord):
        total_coord[i] /= len(joint_data)
    return total_coord

def combine_datasets(rel_data, vel_data, angle_data, images, joints_output, meta = 5):
    '''
    Combines either 2 or 3 data streams into 6-9D co-ordinates in a single dataset file

    Arguments
    ---------
    rel_data, vel_data, angle_data : List(List())
        original joints for 3 separate joint streams

    Returns
    -------
    List(List())
        The single dataset combining the inputted streams
    '''
    print("Combining datasets...")
    rel_data, images = Utilities.process_data_input(rel_data, images)
    vel_data, _ = Utilities.process_data_input(vel_data, None)
    angle_data, _ = Utilities.process_data_input(angle_data, None)
    combined_dataset = []
    for i, row in enumerate(tqdm(rel_data)):
        combined_row = row[0:meta + 1]
        for j, joint in enumerate(row):
            if j > meta:
                if j == meta + 1:
                    avg_joint = avg_coord(row[meta + 2: meta + 9])
                    avg_vel = avg_coord(vel_data[i][meta + 2: meta + 9])
                    if angle_data:
                        avg_ang = avg_coord(angle_data[i][meta + 2: meta + 9])
                    if angle_data:
                        combined_row.append([avg_joint[0], avg_joint[1], avg_joint[2],
                        avg_vel[0], avg_vel[1], avg_vel[2],
                        avg_ang[0], avg_ang[1], avg_ang[2] ])
                    else:
                        combined_row.append([avg_joint[0], avg_joint[1], avg_joint[2],
                        avg_vel[0], avg_vel[1], avg_vel[2]])
                elif j > 10:
                    if angle_data:
                        combined_row.append([joint[0], joint[1], joint[2],
                                        vel_data[i][j][0], vel_data[i][j][1], vel_data[i][j][2],
                                        angle_data[i][j][0], angle_data[i][j][1], angle_data[i][j][2] ])
                    else:
                        combined_row.append([joint[0], joint[1], joint[2],
                                        vel_data[i][j][0], vel_data[i][j][1], vel_data[i][j][2]]) 
        combined_dataset.append(combined_row)
    print("Completing combined dataset.")
    Utilities.save_dataset(combined_dataset, joints_output)
    return combined_dataset

def process_empty_frames(joint_file, image_file, joint_output, image_output):
    '''
    Combined function for loading, processing empty frames for removal and saving the result.

    Arguments
    ---------
    joint_file : List(List()) or str
        Original joints dataset either as a path or the loaded file
    image_file : List(List()) or str
        Corresponding images to the joints_file either as a path or the loaded file
    joint_output : str
        Desired output path for the joints data
    image_output : str
        Desired output path for the image data

    Returns
    -------
    List(List()), List(List())
        Resulting joint and image data

    '''
    print("\nProcessing Empty frames...")
    joint_data, image_data = Utilities.process_data_input(joint_file, image_file, cols=Utilities.colnames, ignore_depth=True)

    joint_data, image_data = Data_Correction.remove_empty_frames(joint_data, image_data)
    Utilities.save_dataset(joint_data, joint_output)
    Utilities.save_images(joint_data, image_data, image_output)
    print("Empty frame processing complete.")

    return joint_data, image_data

def process_trimmed_frames(joint_file, image_file, joint_output, image_output, trim):
    '''
    Creator function to load, trim and then save joint and image folders.

    Arguments
    ---------
    joint_file : List(List()) or str
        Original joints dataset either as a path or the loaded file
    image_file : List(List()) or str
        Corresponding images to the joints_file either as a path or the loaded file
    joint_output : str
        Desired output path for the joints data
    image_output : str
        Desired output path for the image data
    trim : int
        Number of images/joint sequences at the start and end of each sequence to cut

    Returns
    -------
    List(List()), List(List())
        Resulting joint and image data

    '''
    print("\nProcessing trimmed frames...")
    joint_data, image_data = Utilities.process_data_input(joint_file, image_file)
    joint_data, image_data = Data_Correction.trim_frames(joint_data, image_data, trim = trim)
    Utilities.save_dataset(joint_data, joint_output)
    Utilities.save_images(joint_data, image_data, image_output)
    print("Trimmed frame processing complete.")
    return joint_data, image_data
    
def create_relative_dataset(joint_file, image_file, joint_output, meta = 5):
    '''
    Creator function for relativizing absolute joint co-ordinates and saving the result

    Arguments
    ---------
    joint_file : List(List()) or str
        Original joints dataset either as a path or the loaded file
    image_file : List(List()) or str
        Corresponding images to the joints_file either as a path or the loaded file
    joint_output : str
        Desired output path for the joints data
    meta : int (optional, default = 5)
        denotes the amount of metadata to expect in each joint row.

    Returns
    -------
    List(List())
        Resulting joint data

    '''
    print("\nCreating relative value dataset...")
    abs_data, image_data = Utilities.process_data_input(joint_file, image_file)
    rel_data = []
    for i, joints in enumerate(tqdm(abs_data)):
        rel_row = []
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

    Utilities.save_dataset(rel_data, joint_output)
    print("relative dataset completed.")
    return rel_data

def create_normalized_dataset(joint_file, image_file, joint_output):
    '''
    Creator function for passing the joint data through the various normalization and outlier-detection functions.

    Arguments
    ---------
    joint_file : List(List()) or str
        Original joints dataset either as a path or the loaded file
    image_file : List(List()) or str
        Corresponding images to the joints_file either as a path or the loaded file
    joint_output : str
        Desired output path for the joints data

    Returns
    -------
    List(List())
        Resulting joint data

    '''
    print("\nNormalizing joint data...")
    joint_data, image_data = Utilities.process_data_input(joint_file, image_file, cols=Utilities.colnames)
    #Normalize depth extremities
    joint_data = Data_Correction.normalize_outlier_depths(joint_data, image_data)
    #Normalize outlier values
    joint_data = Data_Correction.normalize_outlier_values(joint_data, image_data, 100)
    #Smooth values that are visually a bit haywire on the arms and legs
    joint_data = Data_Correction.smooth_unlikely_values(joint_data, image_data)
    Utilities.save_dataset(joint_data, joint_output)
    print("Data normalization complete.")
    return joint_data

def create_scaled_dataset(joint_file, image_file, joint_output):
    '''
    Creator function for creating a scaled dataset with the bespoke scaler function.

    Arguments
    ---------
    joint_file : List(List()) or str
        Original joints dataset either as a path or the loaded file
    image_file : List(List()) or str
        Corresponding images to the joints_file either as a path or the loaded file
    joint_output : str
        Desired output path for the joints data

    Returns
    -------
    List(List())
        Resulting joint data

    '''
    #Standardize the scale of the human in the data
    print("\nScaling data...")
    joint_data, image_data = Utilities.process_data_input(joint_file, image_file)
    joint_data = Data_Correction.normalize_joint_scales(joint_data, image_data)
    Utilities.save_dataset(joint_data, joint_output)
    print("Data scale processing complete.")
    return joint_data

def create_velocity_dataset(joint_file, image_file, joint_output):
    '''
    Creator function for creating a dataset of motion values.

    Arguments
    ---------
    joint_file : List(List()) or str
        Original joints dataset either as a path or the loaded file
    image_file : List(List()) or str
        Corresponding images to the joints_file either as a path or the loaded file
    joint_output : str
        Desired output path for the joints data

    Returns
    -------
    List(List())
        Resulting joint data

    '''
    print("\nCreating velocities dataset...")
    joint_data, image_data = Utilities.process_data_input(joint_file, image_file)
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

def append_midhip(joint_file, image_file, joint_output):
    '''
    Simple Creator function for adding an artificial mid-hip joint to the dataset.

    Arguments
    ---------
    joint_file : List(List()) or str
        Original joints dataset either as a path or the loaded file
    image_file : List(List()) or str
        Corresponding images to the joints_file either as a path or the loaded file
    joint_output : str
        Desired output path for the joints data

    Returns
    -------
    List(List())
        Resulting joint data

    '''
    abs_data, images = Utilities.process_data_input(joint_file, image_file, cols=Utilities.colnames)
    midhip_dataset = []
    for i, joints in enumerate(abs_data):
        midhip_row = list(joints)
        midhip_row.append(Utilities.midpoint(joints[14], joints[15]))
        midhip_dataset.append(midhip_row)
        #Display results
        #Render.render_joints(images[i], midhip_row, delay=True)
    
    Utilities.save_dataset(midhip_dataset, joint_output, colnames=Utilities.colnames_midhip)
    return midhip_dataset

def create_bone_dataset(joint_file, image_file, joint_output, meta = 6):
    '''
    Creator function for creating a dataset of bone angles.

    Arguments
    ---------
    joint_file : List(List()) or str
        Original joints dataset either as a path or the loaded file
    image_file : List(List()) or str
        Corresponding images to the joints_file either as a path or the loaded file
    joint_output : str
        Desired output path for the joints data
    meta : int
        number of expected metadata values per row
        
    Returns
    -------
    List(List())
        Resulting joint data
    '''
    abs_data, images = Utilities.process_data_input(joint_file, image_file)
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
                    else:
                        tmp_vector = [0,0,0]
                    bone_row.append(tmp_vector)
        bone_dataset.append(bone_row)
        #Render.render_velocities(abs_data[i], bone_row, images[i])
    Utilities.save_dataset(bone_dataset, joint_output, colnames=Utilities.colnames_midhip)
    return bone_dataset                 
    
def create_hcf_dataset(pre_abs_joints, abs_joints, rel_joints, vel_joints, images, joint_output, meta = 5):
    '''
    Experimental Creator function for creating a dataset of hand-crafted features.

    Arguments
    ---------
    pre_abs_joints: str 
        joint data file for absolute co-ordinates, prior to normalization
    abs_joints: str
        joint data file for absolute co-ordinates
    rel_joints: str
        joint data file for relativized co-ordinates
    vel_joints: str
        joint data file for velocity vectors
    image_data: str
        image data folder corresponding to the images
    joint_output : str
        Desired output path for the joints data
    meta : int
        number of expected metadata values per row
        
    Returns
    -------
    List(List())
        Resulting joint data
    '''
    abs_joint_data, images = Utilities.process_data_input(abs_joints, images)
    rel_joint_data, _ =  Utilities.process_data_input(rel_joints, None)
    abs_veljoint_data, _ =  Utilities.process_data_input(vel_joints, None)
    print("Building HCF Dataset...", len(abs_joint_data), len(rel_joint_data), len(pre_abs_joints))
    pre_gait_cycles = hcf.get_gait_cycles(pre_abs_joints, None)
    gait_cycles = Utilities.set_gait_cycles(abs_joint_data, pre_gait_cycles)
    rel_gait_cycles = Utilities.set_gait_cycles(rel_joint_data, pre_gait_cycles)

    print("gait cycle lens: ", len(pre_gait_cycles), len(gait_cycles), len(rel_gait_cycles))
    knee_data_cycles = Utilities.build_knee_joint_data(pre_gait_cycles, images)
    knee_data_coeffs = Render.chart_knee_data(knee_data_cycles, False)
    trend = hcf.get_knee_chart_polynomial(knee_data_cycles)

    feet_heights = hcf.get_feet_height(gait_cycles, images)
    times_LOG, times_not_moving = hcf.get_time_LofG(gait_cycles, abs_veljoint_data, images)
    speeds = hcf.get_speed(gait_cycles, images)

    stride_lengths, stride_ratios = hcf.get_stride_lengths(rel_gait_cycles, images, gait_cycles)
    stride_gaps, max_gaps = hcf.get_stride_gap(gait_cycles, images)
    gait_cycles_dataset = []
    for i, cycle in enumerate(gait_cycles):
        #Add metadata
        hcf_cycle = cycle[0][0:meta + 1]
        hcf_cycle.append(feet_heights[i][0])
        hcf_cycle.append(feet_heights[i][1])
        hcf_cycle.append(times_LOG[i][0])
        hcf_cycle.append(times_LOG[i][1])
        hcf_cycle.append(times_not_moving[i])
        hcf_cycle.append(speeds[i])
        hcf_cycle.append(stride_gaps[i])
        hcf_cycle.append(stride_lengths[i][1])
        hcf_cycle.append(max_gaps[i])
        for c in knee_data_coeffs[i]:
            hcf_cycle.append(c)
        gait_cycles_dataset.append(copy.deepcopy(hcf_cycle))

    if joint_output != None:
        Utilities.save_dataset(gait_cycles_dataset, joint_output, Utilities.hcf_colnames)
    print("HCF dataset completed.")
    return gait_cycles_dataset

#Create dummy frames by applying gaussian noises to the original frames
def create_dummy_dataset(joint_file, joint_output):
    '''
    Creator function for generating dummy data to enlarge original datasets

    Arguments
    ---------
    joint_file : List(List()) or str
        Original joints dataset either as a path or the loaded file
    joint_output : str
        Desired output path for the joints data
        
    Returns
    -------
    List(List())
        Resulting joint data
    '''
    data, _ = Utilities.process_data_input(joint_file, None)
    mean = 0 
    std_dev = 0.1
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
    #Get total number of people
    no_people = sequences[-1][0][5]
    if no_people <= 0:
        no_people =1

    scaling_factors = [0 for i in range(no_people)]
    frame_counts = [0 for i in range(no_people)]
    #Minimum number of examples you want per person in frames, this is taken from the best of the original 15 participants in terms
    #of raw accuracy
    threshold = 13000
    for sequence in sequences:
        for frame in sequence:
            #Add -1 for weightgait
            #Edge case: producing single person datasets sets person to -1
            if frame[5] < 0:
                frame[5] = 0
            #print("what is this: ", frame[5], len(frame_counts), frame_counts)
            frame_counts[frame[5]-1] += 1

    for i, factor in enumerate(scaling_factors):
        #Calculate what I need to multiply by to reach a minimum of 13000 frames (or 2200ish sequences of 6)
        scaling_factors[i] = int(threshold /frame_counts[i])

    scaling_iter = 0
    original_len = len(sequences)
    for i, sequence in enumerate(sequences):
        #First: Add the original frames
        if i > 0 and (i + 1) % 60 == 0:
            scaling_iter +=1
            if scaling_iter > len(scaling_factors)-1:
                scaling_iter = 0

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
    for frame in novel_sequences:
        noise_sequences.append(frame)
    Utilities.save_dataset(noise_sequences, joint_output)
    return noise_sequences
                   
def subtract_skeleton(joint_file, joint_output, base_output):
    '''
    Creator function for generating a file of joint values after being masked by the average gait cycle.

    Arguments
    ---------
    joint_file : List(List()) or str
        Original joints dataset either as a path or the loaded file
    joint_output : str
        Desired output path for the joints data
        
    Returns
    -------
    List(List())
        Resulting joint data
    '''
    rel_sequences = Utilities.convert_to_sequences(joint_file)
    rel_sequences = Utilities.interpolate_gait_cycle(rel_sequences, base_output, 0, restrict_cycle=False) 

    #Extract an overlay sequence for each individual
    overlay_sequences = [s for i, s in enumerate(rel_sequences) if i % 60 == 0]
    for i, sequence in enumerate(rel_sequences):
        if i % 60 == 0:
            overlay_sequences.append(Utilities.get_average_sequence(rel_sequences[i:i+10]))
    #Subtract the data by the mask
    overlay_iter = 0
    sequence_counter = 0
    for i, sequence in enumerate(rel_sequences):
        for j, frame in enumerate(sequence):
            for k, coord in enumerate (frame):
                if k> 5:
                    #Check if coord and overlay[j][k] are within a radius of eachother, ignoring the first 10
                    try:
                        if Utilities.check_within_radius(coord, overlay_sequences[overlay_iter][j][k], 10):
                            #print("detected within raidus: ", coord, overlay_sequence[j][k])
                            rel_sequences[i][j][k] = [0.0, 0.0, 0.0]
                            #nothing = 0
                    except:
                        pass
        if i % 60 == 0 and i != 0:
            overlay_iter += 1
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
    '''
    Creator function for converting from 3-class to 9 class files

    Arguments
    ---------
    joint_file : List(List()) or str
        Original joints dataset either as a path or the loaded file
    joint_output : str
        Desired output path for the joints data
        
    Returns
    -------
    List(List())
        Resulting joint data
    '''
    for i, row in enumerate(data):
        if row[2] == 0:
            #Freeze 0
            if row[3] == 0:
                #Obstacle 0
                if row[4] == 0:
                    #Class 0, freeze 0, obstacle 0 = 0
                    data[i][2] = 0
                else:
                    #Class 0, freeze 0, obstacle 0 = 1
                    data[i][2] = 2
            else:
                data[i][2] = 1
        elif row[2] == 1:
            #Freeze 0
            if row[3] == 0:
                #Obstacle 0
                if row[4] == 0:
                    #Class 1, freeze 0, obstacle 0 = 0
                    data[i][2] = 3
                else:
                    #Class 1, freeze 0, obstacle 0 = 1
                    data[i][2] = 5
            else:
                data[i][2] = 4
        elif row[2] == 2:
            #Freeze 0
            if row[3] == 0:
                #Obstacle 0
                if row[4] == 0:
                    #Class 0, freeze 0, obstacle 0 = 0
                    data[i][2] = 6
                else:
                    #Class 0, freeze 0, obstacle 0 = 1
                    data[i][2] = 8
            else:
                data[i][2] = 7
    Utilities.save_dataset(data, joint_output)
    return data
