import Programs.Data_Processing.Model_Based.Data_Correction as Data_Correction
import Programs.Data_Processing.Model_Based.Utilities as Utilities
import Programs.Data_Processing.Model_Based.Render as Render

def assign_class_labels(num_switches, num_classes, joint_file, joint_output):
    joint_data = Utilities.load(joint_file)
    joint_data = Utilities.apply_class_labels(num_switches, num_classes, joint_data)
    Utilities.save_dataset(joint_data, joint_output)
    return joint_data

def process_empty_frames(joint_file, image_file, joint_output, image_output):
    joint_data, image_data = Utilities.process_data_input(joint_file, image_file)
    print("loading in data: ", len(joint_data), len(image_data))
    joint_data, image_data = Data_Correction.remove_empty_frames(joint_data, image_data)
    print("full frame count: ", len(image_data), len(joint_data))
    Utilities.save_dataset(joint_data, joint_output)
    Utilities.save_images(joint_data, image_data, image_output)

    return joint_data, image_data

def process_trimmed_frames(joint_file, image_file, joint_output, image_output, trim):
    joint_data, image_data = Utilities.process_data_input(joint_file, image_file)
    joint_data, image_data = Data_Correction.trim_frames(joint_data, image_data, trim = trim)
    Utilities.save_dataset(joint_data, joint_output)
    Utilities.save_images(joint_data, image_data, image_output)
    return joint_data, image_data

    
def create_relative_dataset(abs_data, image_data, joint_output):
    abs_data, image_data = Utilities.process_data_input(abs_data, image_data)
    rel_data = []
    for i, joints in enumerate(abs_data):
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
    return rel_data

def create_normalized_dataset(joint_data, image_data, joint_output):
    joint_data, image_data = Utilities.process_data_input(joint_data, image_data)
    #Normalize depth extremities
    joint_data = Data_Correction.normalize_outlier_depths(joint_data, image_data)
    #Normalize outlier values
    joint_data = Data_Correction.normalize_outlier_values(joint_data, image_data, 100)
    Utilities.save_dataset(joint_data, joint_output)
    return joint_data

def create_scaled_dataset(joint_data, image_data, joint_output):
    #Standardize the scale of the human in the data
    joint_data, image_data = Utilities.process_data_input(joint_data, image_data)
    joint_data = Data_Correction.normalize_joint_scales(joint_data, image_data)
    Utilities.save_dataset(joint_data, joint_output)
    return joint_data

def create_flipped_joint_dataset(rel_data, abs_data, images, joint_output):
    abs_data, images = Utilities.process_data_input(abs_data, images)
    rel_data, _ = Utilities.process_data_input(rel_data, images)

    sequence_data = Utilities.convert_to_sequences(abs_data)
    rel_sequence_data = Utilities.generate_relative_sequence_data(sequence_data, rel_data)

    flipped_data = [] 
    total = 0 
    for i, seq in enumerate(rel_sequence_data):
        print("running total: ", len(seq), total)
        total += len(seq)

    print("sequence total:", total)
    for i, seq in enumerate(rel_sequence_data):

        #Get first and last head positions (absolute values)
        first = sequence_data[i][1]
        last = sequence_data[i][-1]

        #This is going from right to left: the ones we want to flip
        print("first and last: ", first[3][0], last[3][0])
        if first[3][1] > last[3][1]:
            print("flipping joint sequence: ", i)
            for joints in seq:
                #Append with metadata
                flipped_joints = [joints[0], joints[1], joints[2]]
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
    #for k, joints in enumerate(flipped_data):
    #    if k > 20 and k < 50:
    #        #Render.render_joints(images[k], flipped_data[k], delay=True)
    #        Render.plot3D_joints(flipped_data[k], x_rot=90, y_rot=0)
    #    elif k > 50:
    #        break

    Utilities.save_dataset(flipped_data, joint_output)
    return flipped_data

def create_velocity_dataset(joint_data, image_data, joint_output):
    joint_data, image_data = Utilities.process_data_input(joint_data, image_data)
    velocity_dataset = []
    for i, joints in enumerate(joint_data):
        if i+1 < len(joint_data) and i > 0:
            velocity_dataset.append(Utilities.plot_velocity_vectors(image_data[i], joint_data[i - 1], joints, joint_data[i + 1]))
        elif i+1 < len(joint_data) and i <= 0:
            velocity_dataset.append(Utilities.plot_velocity_vectors(image_data[i], [0], joints, joint_data[i + 1]))
        elif i+1 >= len(joint_data) and i > 0:
            velocity_dataset.append(Utilities.plot_velocity_vectors(image_data[i], joint_data[i - 1], joints, [0]))
    
    Utilities.save_dataset(velocity_dataset, joint_output)
    return velocity_dataset


def create_joint_angle_dataset():
    pass

def create_hcf_dataset():
    pass

def create_regions_dataset():
    pass