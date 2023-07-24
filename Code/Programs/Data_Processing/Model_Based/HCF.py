import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import Programs.Data_Processing.Model_Based.Render as Render


def split_by_instance(joint_data, pad = True):

    #Find longest instance first:
    max_instance_length = 0
    for row in joint_data:
        if row[1] > max_instance_length:
            max_instance_length = row[1]

    print("max length: ", max_instance_length)

    #set padded sequences to 40 unless there's examples that are bigger, otherwise set it to that.
    std_inst_length = 40
    if max_instance_length >= std_inst_length:
        std_inst_length = max_instance_length

    gait_cycles = []
    current_cycle = []
    current_instance = 0
    for row in joint_data:

        if row[0] == current_instance:
            current_cycle.append(row)
        else:
            current_instance += 1
            if len(current_cycle) < std_inst_length:
                zero_row = copy.deepcopy(current_cycle[-1])
                for i, c in enumerate(zero_row):
                    if i > 5:
                        zero_row[i] = list(np.zeros(len(c)))
                
                if pad:
                    while len(current_cycle) < std_inst_length:
                        current_cycle.append(zero_row)

                #Appending gait cycle
                gait_cycles.append(copy.deepcopy(current_cycle))
                current_cycle = []
    
    #Appending final gait cycle
    while len(current_cycle) < std_inst_length:
        current_cycle.append(zero_row)
    
    gait_cycles.append(copy.deepcopy(current_cycle))

    return gait_cycles

def get_knee_chart_polynomial(data):
    trends = []
    for i in range(len(data[0])):
        trend = np.polyfit(data[0][i], data[1][i], 3)
        plt.plot(data[0][i],data[1][i],'o')
        trendpoly = np.poly1d(trend) 
        plt.plot(data[0][i],trendpoly(data[0][i]))
        plt.show()
        trends.append(trend)

    #This returns 4 polynomial co-efficients that approximate the knee angle curve.
    return trends

def get_stride_gap(gait_cycles, images):
    stride_gaps = []
    biggest_gaps = []
    image_iter = 0
    for i, frame in enumerate(gait_cycles):
        gaps_in_frames = []
        max_gap = 0
        #Get distance between the feet in every frame and check if that's this cycle's maximum
        for j, joints in enumerate(frame):
            gap = math.dist(joints[22], joints[21])
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
            relative_stride_0 = math.dist(joints[21], joints[17])
            relative_stride_1 = math.dist(joints[22], joints[18])
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
        first = cycle[1]
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
            print("values: ", image_iter, len(velocity_joints[image_iter]))
            left_velocity = abs(velocity_joints[image_iter][21][0]) + abs(velocity_joints[image_iter][21][1])+ abs(velocity_joints[image_iter][21][2])\
                + abs(velocity_joints[image_iter][17][0])+ abs(velocity_joints[image_iter][17][1]+ abs(velocity_joints[image_iter][17][2]))
            right_velocity = abs(velocity_joints[image_iter][22][0]) + abs(velocity_joints[image_iter][22][1])+ abs(velocity_joints[image_iter][22][2])\
                + abs(velocity_joints[image_iter][18][0])+ abs(velocity_joints[image_iter][18][1]+ abs(velocity_joints[image_iter][18][2]))


            #print("velocities: ", left_velocity, right_velocity)
            #Left leg moving, leg is off ground
            if left_velocity > right_velocity and left_velocity >= right_velocity + threshold:
                frames_off_ground[0] += 1
                #print("left leg off ground")
                #render_joints(images[image_iter], joints, delay=True, use_depth=False, colour=(0,0, 255))
            #Right leg moving
            elif right_velocity > left_velocity and right_velocity >= left_velocity + threshold:
                frames_off_ground[1] += 1
                #print("right leg off ground")
                #render_joints(images[image_iter], joints, delay=True, use_depth=False, colour=(0,0, 255))
            #Neither leg moving, this is double support
            else:
                #print("neither leg off ground")
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
            total_feet_height[0] += math.dist(joints[21], joints[17])
            total_feet_height[1] += math.dist(joints[22], joints[18])
            
            #print("feet heights: ", feet_heights, total_feet_height, len(gait_cycles))
            #render_joints(images[image_iter], joints, delay=True, use_depth=False, colour=(255,0, 0))
            image_iter += 1

        #Take the average height across the gait cycle
        average_feet_height = [0,0]
        average_feet_height[0] = total_feet_height[0] / len(gait_cycles[i])
        average_feet_height[1] = total_feet_height[1] / len(gait_cycles[i])
        feet_heights.append(copy.deepcopy(average_feet_height))

    return feet_heights