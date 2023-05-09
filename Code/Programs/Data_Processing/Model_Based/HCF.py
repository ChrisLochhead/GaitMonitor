import numpy as np
import csv
import copy
import math
import numpy as np
from Programs.Data_Processing.Model_Based.Utilities import load, load_images
from Programs.Data_Processing.Model_Based.Render import render_joints, joint_connections
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D   
from scipy.interpolate import make_interp_spline, BSpline

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]

def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360
    
    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else: 
        
        return ang_deg
    
def build_knee_joint_data(gait_cycle):
    l_angles = []
    r_angles = []

    for i, frame in enumerate(gait_cycle):
        #Left hip to left knee, left knee to left foot
        lh_lk = [frame[14], frame[16]]
        lk_lf = [frame[16], frame[18]]

        #Right hip to right knee, right knee to right foot
        rh_rk = [frame[15], frame[17]]
        rk_rf = [frame[17], frame[19]]

        l_angles.append(ang(lh_lk, lk_lf))
        r_angles.append(ang(rh_rk, rk_rf))

    return [l_angles, r_angles]


def interpolate_knee_data(x, y, scale = 5000):
    curr_length = len(x)
    inter_length = (curr_length -1) * scale
    inter_data = []
    inter_indices = [i for i in range(inter_length + curr_length)]
    for i, instance in enumerate(x):
        #print("appending initial: ", instance)
        inter_data.append(instance)
        #Don't do it for final instance
        if i < len(x) - 1:
            angle_change = abs(instance - x[i + 1])
            #inter_changes = np.logspace(np.log(x[i+1]), np.log(instance), scale, base=np.exp(1))

            for j in range(1, scale + 1):
                #print("interpolating from ", instance, " to ", x[i + 1])
                #print("current: ", j, " of 4. Value is: ", inter_changes[j-1])
                if instance < x[i+1]:
                    inter_data.append(instance + ((angle_change / scale) * j))
                    #print("added value: ", instance + ((angle_change/scale) * j))
                else:
                    inter_data.append(instance - ((angle_change / scale) * j))
                    #print("added value: ", instance - ((angle_change/scale) * j))
    
    print("final lens: ", len(inter_data), len(inter_indices))
    return inter_data, inter_indices

def chart_knee_data(gait_cycle_angles):
    #fig = plt.figure()
    #ax = fig.add_subplot(111)


    l_x = gait_cycle_angles[0]
    l_y = [i for i in range(len(l_x))]

    r_x = gait_cycle_angles[1]
    r_y = [i for i in range(len(r_x))]

    print("length x: ", len(l_x), len(l_y), l_y)

    #Potentially add interpolation code here to give more examples for a smoother chart
    l_x, l_y = interpolate_knee_data(l_x, l_y)
    r_x, r_y = interpolate_knee_data(r_x, r_y)

    #print("showing original")
    #plt.figure()
    #plt.plot(l_y,l_x)
    #plt.plot(r_y,r_x)
    #plt.show()


    print("showing poly")
    plt.figure()
    poly = np.polyfit(l_y,l_x,8)
    poly_alt = np.polyfit(r_y, r_x, 8)
    poly_l = np.poly1d(poly)(l_y)
    poly_r = np.poly1d(poly_alt)(r_y)
    plt.plot(l_y,poly_l)
    plt.plot(r_y,poly_r)
    plt.show()


#Get cycles in each instance of 5 frames each
def get_time_cycles(joint_data, images):
    instances = []
    instance = []
    threshold = 5
    current_instance = 0
    for row_iter, row in enumerate(joint_data):
        current_instance = row[0]
        instance.append(row)
        if len(instance) >= threshold:
            instances.append(copy.deepcopy(instance))
            instance = []
            continue 

        if len(joint_data) > row_iter + 1:
            if joint_data[row_iter + 1][0] != current_instance:
                #Add to previous to make sure no cycle is less than 5
                if len(instance) < threshold:
                    for inst in instance:
                        instances[-1].append(inst)
                instance = []
                continue
    return instances
        


#This will only work with relative data
def get_gait_cycles(joint_data, images):
    instances = []
    instance = []
    current_instance = 0
    #First separate the joints into individual instances
    for joints in joint_data:
        #Append joints as normal
        if joints[0] == current_instance:
            instance.append(joints)
        else:
            #If this is the first of a new instance, add the old instance to the array,
            #clear it, add this current one and reset what current instance is.
            instances.append(copy.deepcopy(instance))
            instance = []
            instance.append(joints)
            current_instance = joints[0]

    gait_cycles = []
    gait_cycle = []

    #For each instance
    for i, inst in enumerate(instances):
        found_initial_direction = False
        direction = -1
        crossovers = 0

        for j, row in enumerate(inst):
            #Only register initial direction if they aren't equal
            if found_initial_direction == False:
                if row[18][1] > row[19][1]:
                    direction = 0
                    found_initial_direction = True
                elif row[18][1] < row[19][1]:
                    direction = 1
                    found_initial_direction = True

            #Check if the direction matches the current movement and append as appropriate
            if row[18][1] > row[19][1] and direction == 0:
                gait_cycle.append(row)
            elif row[18][1] < row[19][1] and direction == 1:
                gait_cycle.append(row)
            elif row[18][1] > row[19][1] and direction == 1:
                crossovers += 1
                print("crossover detected")
                gait_cycle.append(row)
                direction = 0
            elif row[18][1] < row[19][1] and direction == 0:
                crossovers += 1
                print("crossover detected")
                gait_cycle.append(row)
                direction = 1
            else:
                #There is either no cross over or the rows are totally equal, in which case just add as usual
                gait_cycle.append(row)

            #render_joints(images[j], row, delay=True, use_depth=False)

            #Check the number of crossovers 
            #If there has been 2 this is one full gait cycle, append it to the gait cycles and reset the 
            #current gait cycle.
            if crossovers > 1:
                crossovers = 0
                gait_cycles.append(copy.deepcopy(gait_cycle))
                gait_cycle = []

            #Check if we are at the end of the instance and adjust the last cycle accordingly so we dont end up with length 1 gait cycles.
            if j == len(inst):
                #If the current gait cycle isn't at least 5 frames, just append it on to the latest one
                if len(gait_cycle) < 5:
                    for g in gait_cycle:
                        gait_cycles[-1].append(g)
                    gait_cycles = []
                else:
                #Otherwise just add this one and reset it before the next instance starts.
                    gait_cycles.append(gait_cycle)
                    gait_cycle = []

    #Illustrate results
    col = (0,0,255)
    image_iter = 0
    for cycle in gait_cycles:
        #Switch the cycle every new gait cycle
        if col == (0,0,255):
            col = (255,0,0)
        else:
            col = (0,0,255)
        for i, row in enumerate(cycle):
            #Render every frame
            print("frame ", i, " of ", len(cycle))
            #render_joints(images[image_iter], row, delay=True, use_depth=False, colour=col)
            image_iter += 1
            
    return gait_cycles


def get_stride_gap(gait_cycles, images):
    stride_gaps = []
    biggest_gaps = []
    image_iter = 0
    for i, frame in enumerate(gait_cycles):
        gaps_in_frames = []
        max_gap = 0
        #Get distance between the feet in every frame and check if that's this cycle's maximum
        for j, joints in enumerate(frame):
            gap = math.dist(joints[19], joints[18])
            gaps_in_frames.append(gap)
            if gap > max_gap:
                max_gap = gap
            #render_joints(images[image_iter], gait_cycles[i][j], delay=True)
            image_iter += 1
        #Get average gap during the gait cycle
        biggest_gaps.append(max_gap)
        stride_gaps.append(sum(gaps_in_frames)/len(gaps_in_frames))

    return stride_gaps, biggest_gaps

def get_stride_lengths(rel_gait_cycles, images, gait_cycles):
    stride_lengths = []
    stride_ratios = []
    image_iter = 0

    for i, frame in enumerate(gait_cycles):
        max_stride_lengths =[0,0]
        stride_ratio = 0
        #Get max stride length values in this cycle
        for j, joints in enumerate(frame):
            relative_stride_0 = math.dist(joints[18], joints[14])
            relative_stride_1 = math.dist(joints[19], joints[15])
            if relative_stride_0 > max_stride_lengths[0]:
                max_stride_lengths[0] = copy.deepcopy(relative_stride_0)
            if relative_stride_1  > max_stride_lengths[1]:
                max_stride_lengths[1] = copy.deepcopy(relative_stride_1)
            #render_joints(images[image_iter], gait_cycles[i][j], delay=True)
            image_iter += 1
        stride_lengths.append(max_stride_lengths)
        stride_ratio = max_stride_lengths[0]/max_stride_lengths[1]
        stride_ratios.append(stride_ratio)

    return stride_lengths, stride_ratios
            

def get_speed(gait_cycles, images):
       
    speeds = []
    for i, cycle in enumerate(gait_cycles):
        speed = 0
        #Sometimes first frame can be messy, get second one.
        first = cycle[0]
        #Get last frame (some last ones corrupted, get third from last)
        last = cycle[-1]
        #Get the average speed throughout the frames
        speed = np.sqrt((abs(last[0] - first[0])**2) + (abs(last[1] - first[1])**2) + (abs(last[2] - first[2])**2))
        speeds.append(speed/len(cycle))
    return speeds

def get_time_LofG(gait_cycles, velocity_joints, images):
    frames_off_ground_array = []
    both_not_moving_array = []
    threshold = 0.55
    image_iter = 0
    for i, frame in enumerate(gait_cycles):
        frames_off_ground = [0,0]
        frames_not_moving = 0
        for j, joints in enumerate(frame):
            #Calculate relative velocities between hips and feet
            left_velocity = abs(velocity_joints[image_iter][18][0]) + abs(velocity_joints[image_iter][18][1])+ abs(velocity_joints[image_iter][18][2])\
                + abs(velocity_joints[image_iter][14][0])+ abs(velocity_joints[image_iter][14][1]+ abs(velocity_joints[image_iter][14][2]))
            right_velocity = abs(velocity_joints[image_iter][19][0]) + abs(velocity_joints[image_iter][19][1])+ abs(velocity_joints[image_iter][19][2])\
                + abs(velocity_joints[image_iter][15][0])+ abs(velocity_joints[image_iter][15][1]+ abs(velocity_joints[image_iter][15][2]))


            print("velocities: ", left_velocity, right_velocity)
            #Left leg moving, leg is off ground
            if left_velocity > right_velocity and left_velocity >= right_velocity + threshold:
                frames_off_ground[0] += 1
                print("left leg off ground")
                #render_joints(images[image_iter], joints, delay=True, use_depth=False, colour=(0,0, 255))
            #Right leg moving
            elif right_velocity > left_velocity and right_velocity >= left_velocity + threshold:
                frames_off_ground[1] += 1
                print("right leg off ground")
                #render_joints(images[image_iter], joints, delay=True, use_depth=False, colour=(0,0, 255))
            #Neither leg moving, this is double support
            else:
                print("neither leg off ground")
                frames_not_moving += 1
                #render_joints(images[image_iter], joints, delay=True, use_depth=False, colour=(0,0, 255))
           
            image_iter += 1

        frames_off_ground_array.append(copy.deepcopy(frames_off_ground))
        both_not_moving_array.append(frames_not_moving)
    
    return frames_off_ground_array, both_not_moving_array
            

def get_feet_height(gait_cycles, images):
    feet_heights = []  
    image_iter = 0
    for i, frame in enumerate(gait_cycles):
        total_feet_height = [0,0]
        for j, joints in enumerate(frame):
            #Illustrate feet height by gap between hip and foot to indicate
            #changing height from the ground
            total_feet_height[0] += math.dist(joints[18], joints[14])
            total_feet_height[1] += math.dist(joints[19], joints[15])
            
            #print("feet heights: ", feet_heights, total_feet_height, len(gait_cycles))
            #render_joints(images[image_iter], joints, delay=True, use_depth=False, colour=(255,0, 0))
            image_iter += 1

        #Take the average height across the gait cycle
        average_feet_height = [0,0]
        average_feet_height[0] = total_feet_height[0] / len(gait_cycles[i])
        average_feet_height[1] = total_feet_height[1] / len(gait_cycles[i])
        feet_heights.append(copy.deepcopy(average_feet_height))

    return feet_heights



def create_hcf_dataset(jointfile, rel_jointfile, abs_veljointfile, folder, save = True):
    abs_joint_data = load(jointfile)
    rel_joint_data = load(rel_jointfile)
    abs_veljoint_data = load(abs_veljointfile)
    images = load_images(folder, ignore_depth=False)

    print("images and joints loaded, getting gait cycles...")
    #Gait cycle has instances of rows: each instance contains an array of rows
    #denoting all of the frames in their respective gait cycle, appended with their
    #metadata at the start
    gait_cycles = get_gait_cycles(abs_joint_data, images)

    #Experiment with getting and then charting knee data
    for i in range(len(gait_cycles)):
        if i <= 2:
            angles = build_knee_joint_data(gait_cycles[i])
            chart_knee_data(angles)
    return
    #Terrible
    #gait_cycles = get_time_cycles(abs_joint_data, images)

    #Replicate this pattern for the relative gait cycle (if youn run it directly, 
    #relative gait will return very marginally different gait cycles.)
    rel_gait_cycles = []
    joint_counter = 0
    for cycle in gait_cycles:
        new_cycle = []
        for frame in cycle:
            new_cycle.append(rel_joint_data[joint_counter])
            joint_counter += 1
        rel_gait_cycles.append(copy.deepcopy(new_cycle))

    print("number of total gait cycles: ", len(gait_cycles))
    #Then for every gait cycle create a new instance with the following features: 
    #Cadences returns a scalar for every gait cycle returning the number of steps 
    #cadences = get_cadence(gait_cycles, images)

    #Height of feet above the ground
    #returns feet heights using distance between each foot joint and head in relative
    #terms using absolute data. returned as array of 2 element lists for each foot
    feet_heights = get_feet_height(gait_cycles, images)


    #Time leg off of ground + time both feet not moving
    #if foot velocity > 0 + threshold and math.dist(foot, head) > threshold then foot is in motion, add to vector
    #once this no longer holds, change colour and show
    #implement for left and right leg
    times_LOG, times_not_moving = get_time_LofG(gait_cycles, abs_veljoint_data, images)

    #Speed
    #get absolute values of the head at first frame vs last, divide number of frames by that distance
    speeds = get_speed(gait_cycles, images)

    #Stride length + variance
    #get max and min from left leg relative to head
    #get corresponding values in absolute values for leg
    #math.dist to get the distance between these two values
    #repeat for both legs
    stride_lengths, stride_ratios = get_stride_lengths(rel_gait_cycles, images, gait_cycles)
    stride_gaps, max_gaps = get_stride_gap(gait_cycles, images)
    #Combine all hand crafted features into one concrete dataset, save and return it.
    gait_cycles_dataset = []
    for i, cycle in enumerate(gait_cycles):
        #Add metadata
        hcf_cycle = [cycle[0][0], cycle[0][1], cycle[0][2]]
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

    if save:
        with open("./EDA/Finished_Data/hcf_dataset_pixels.csv","w+", newline='') as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(gait_cycles_dataset)

    return gait_cycles_dataset

