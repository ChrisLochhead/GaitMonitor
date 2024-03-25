'''
This file contains all functions related to generating hand-crafted features.
'''
#imports
import copy
import math
import numpy as np
import random
import matplotlib.pyplot as plt

def get_knee_chart_polynomial(data):
    '''
    Plots a knee-angle chart based on a polynomial of the knee motion over the course of the gait cycle

    Arguments
    ---------
    data: List(List())
        List of joints for a single sequence
       
    Returns
    -------
    List(List)
        Returns the subtracted and dummied datasets
    '''
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
    '''
    Creates a column of gaps between the two feet at every frame in an instance

    Arguments
    ---------
    gait cycles: List(List())
        List of joints arranged by gait cycle
    images: List(List())
        List of corresponding images for debugging
       
    Returns
    -------
    List(), List()
        Returns the stride gaps in every frame and the maximum gaps per cycle
    '''
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

def get_stride_lengths(gait_cycles):
    '''
    Calcutates the lengths of each leg stride per gait cycle

    Arguments
    ---------
    gait_cycles: List(List())
        List of joints for a single sequence
       
    Returns
    -------
    List(), List()
        Returns the stride lengths per-leg and ratios
    '''
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
            
def get_speed(gait_cycles):
    '''
    Retrieves the average speed per gait cycle

    Arguments
    ---------
    gait_cycles: List(List())
        List of joints for a single sequence
       
    Returns
    -------
    List()
        Returns a list of speeds per gait cycle
    '''
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
    '''
    Retrieves the amount of time per gait cycle each leg isnt touching the ground

    Arguments
    ---------
    gait_cycles: List(List())
        List of joints for a single sequence
    velocity_joints: List(List())
        List of the corresponding velocities to each gait cycle
    images: List(List())
        Corresponding images for debugging
       
    Returns
    -------
    List(), List()
        Returns a column of average time off ground and a separate list for time neither foot is moving
    '''
    frames_off_ground_array = []
    both_not_moving_array = []
    threshold = 0.55
    image_iter = 0
    for i, frame in enumerate(gait_cycles):
        frames_off_ground = [0,0]
        frames_not_moving = 0
        for j, joints in enumerate(frame):
            #Calculate relative velocities between hips and feet
            #print("values: ", image_iter, len(velocity_joints[image_iter]))
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
    '''
    Retrieves the feet height per frame then gets the average

    Arguments
    ---------
    gait_cycles: List(List())
        List of joints for a single sequence
    images: List(List())
        Corresponding images for debugging
       
    Returns
    -------
    List()
        Returns a list of foot heights per cycle
    '''
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

def get_gait_cycles(joint_data, images):
    '''
    Retrieves the gait cycles by detecting foot-crossings

    Arguments
    ---------
    joint_data: List(List())
        Entire joint file to be segmented into gait cycles
    images: List(List())
        Images for debugging
       
    Returns
    -------
    List()
        Returns a list of joints segmented into gait cycles
    '''
    instances = []
    instance = []
    current_instance = 0
    #First separate the joints into individual instances
    for joints in joint_data:
        #Append joints as normal
        if joints[0] == current_instance:
            instance.append(joints)
        else:
            #Only add certain persons examples
            instances.append(copy.deepcopy(instance))
            instance = []
            instance.append(joints)
            current_instance = joints[0]
    #Add the last instance hanging off missed by the loop
    instances.append(copy.deepcopy(instance))        

    t = 0
    for d in instances:
        t += len(d)
    gait_cycles = []
    gait_cycle = []
    #For each instance
    for i, inst in enumerate(instances):
        found_initial_direction = False
        direction = -1
        crossovers = 0
        row_18_previous = -1
        for j, row in enumerate(inst):
            #Only register initial direction if they aren't equal
            if found_initial_direction == False:
                if row[-2][1] > row[-1][1]:
                    direction = 0
                    found_initial_direction = True
                elif row[-2][1] < row[-1][1]:
                    direction = 1
                    found_initial_direction = True
            #Check for legs mixing up in frame, only need to check one leg
            #print("gap: ", abs(row_18_previous - row[-2][1]), row_18_previous, row[-1][1])
            if abs(row_18_previous - row[-2][1]) > 50 and found_initial_direction:
                gait_cycle.append(row)
                #print("detected leg switch", j)
                row_18_previous = row[-2][1]
                #Render.render_joints(images[j], row, delay=True, use_depth=False)
                continue
            row_18_previous = row[-2][1]
            #Check if the direction matches the current movement and append as appropriate
            if row[-2][1] > row[-3][1] and direction == 0:
                gait_cycle.append(row)
            elif row[-2][1] < row[-3][1] and direction == 1:
                gait_cycle.append(row)
            elif row[-2][1] > row[-3][1] and direction == 1:
                crossovers += 1
                #print("crossover detected a ", row[18][1], row[19][1], direction )
                gait_cycle.append(row)
                direction = 0
            elif row[-2][1] < row[-3][1] and direction == 0:
                crossovers += 1
                #print("crossover detected b ", row[18][1], row[19][1], direction)
                gait_cycle.append(row)
                direction = 1
            else:
                #There is either no cross over or the rows are totally equal, in which case just add as usual
                gait_cycle.append(row)
            #row 16 is right leg at crossover point
            #print("crossover count: {} and current instance length: {}, direction: {}, row 18: {}, row 19: {} ".format(crossovers, len(gait_cycle), direction, row[18], row[19]))
            #Render.render_joints(images[j], row, delay=True, use_depth=False)

            #Check the number of crossovers 
            #If there has been 2 this is one full gait cycle, append it to the gait cycles and reset the 
            #current gait cycle.
            if crossovers > 1 and len(gait_cycle) > 5 or len(gait_cycle) > 19:
                crossovers = 0
                gait_cycles.append(copy.deepcopy(gait_cycle))
                gait_cycle = []
            if len(gait_cycle) > 19:
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

    #Append final gait cycle missed by loop
    if len(gait_cycle) > 0:
        gait_cycles.append(gait_cycle)

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
            #if len(cycle) > 21:
            #    print("frame ", i, " of ", len(cycle))
            #    print("row: ", row)
            #    Render.render_joints(images[image_iter], row, delay=True, use_depth=False, colour=col)
            image_iter += 1

    return gait_cycles


def sample_gait_cycles(data_cycles, num_classes = 9, class_loc = 2):
    '''
    Decimates the gait cycles so they are all of equal length

    Arguments
    ---------
    data_cycles: List(List())
        Joints segmented by gait cycle

    Returns
    -------
    List(List())
        Decimated gait cycles
    '''
    # Find the length of the biggest sublist
    cycles = [[] for i in range(num_classes)]
    print("len gait cycles: ", len(data_cycles))
    for cycle in data_cycles:
        cycles[cycle[0][class_loc]].append(cycle)
    for c in cycles:
        print("cycle len: ", len(c))
    #done = 5/0
    min_length = min(len(sublist) for sublist in cycles)

    for i, cycle in enumerate(cycles):
        #cycles[i] = cycles[i][0:min_length]
        cycles[i] = random.sample(cycles[i], min_length)
    print("lens 2: ", len(cycles[0]),len(cycles[1]),len(cycles[2]))   
    #Agglomerate into one list:
    gait_cycles = []
    for i, lst in enumerate(cycles):
        for j, cycle in enumerate(lst):
            gait_cycles.append(cycle)
    return gait_cycles