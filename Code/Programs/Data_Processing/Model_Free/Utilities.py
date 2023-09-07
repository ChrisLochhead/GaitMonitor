#Standard
import cv2
import re
from PIL import Image
import numpy as np
import os
import sys
import copy
import pandas as pd
from pathlib import Path
import matplotlib
from scipy import stats

#Local files
import Programs.Data_Recording.JetsonYolo_Main.models.JetsonYolo as JetsonYolo
#import ImageProcessor
#import GEI
import torch
import Programs.Machine_Learning.Model_Free.Resnet.Ensemble as Ensemble
import Programs.Machine_Learning.Model_Free.Resnet.LocalResnet as LocalResnet

#Torch and SKlearn
from torchvision.transforms import ToTensor, Lambda

from statsmodels.stats.contingency_tables import mcnemar

#Only works on the PC version of this app, Jetson doesn't support python 3.7
if sys.version_info[:3] > (3, 7, 0):
    import Programs.Machine_Learning.Model_Free.Mask_RCNN.samples.maskcnn as maskcnn
    
#For printing matplotlib charts from command prompt
matplotlib.use('TkAgg')

########## Utility functions ####################
#################################################

def compare_evaluation(models, testing_data):
    model_predictions = []
    for model in models:
        # Calculate precision and recall
        prediction_array = []
        truths = []

        with torch.no_grad():
            for x, y in testing_data:
                x = x.to(device=LocalResnet.my_device)
                y = y.to(device=LocalResnet.my_device)
                truths.append(y.item())
                scores = model(x)

                # Get prediction probabilities
                probs = torch.nn.functional.softmax(scores, dim=1)
                top_p, top_class = probs.topk(1, dim=1)
                _, predictions = scores.max(1)
                # Iterate through results to get correct or wrong
                zipped = zip(y, predictions)
                for i, j in zipped:
                    if i == j:
                        prediction_array.append(1)
                    else:
                        prediction_array.append(0)


        model_predictions.append(prediction_array)
    print("contingency table: ", len(model_predictions[0]), len(model_predictions[1]))

    contingency_table = [[0,0],[0,0]]
    #Sort into contingency table from raw data
    for x, y in zip(model_predictions[0], model_predictions[1]):
        if x == y and x == 1:
            contingency_table[0][0] += 1
        elif x == y and x == 0:
            contingency_table[1][1] += 1
        elif x != y and x == 1:
            contingency_table[1][0] += 1
        elif x !=y and x == 0:
            contingency_table[0][1] += 1

    print("contingency table: ")
    print(contingency_table)
    return contingency_table

def create_contingency_table(classifier1, classifier2):
    #Models for testing
    model1 = Ensemble.load_model(classifier1)
    model2 = Ensemble.load_model(classifier2)

    #Testing data
    batch_size = 50
    epoch = 5
    target = Lambda( lambda y: torch.zeros(2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    training, testing = Ensemble.split_data_n_folds(num_folds=3,
                                             sourceTransform=ToTensor(),
                                             targetTransform=target,
                                             sizes='./Instance_Counts/FewShot/Normal/indices.csv',
                                             # <- change this between GEI or FFGEI/HOGFFGEI and graphcut
                                             batch_size=batch_size,
                                             FFGEI=False,
                                             data_path='./Images/FFGEI/FewShot/Unravelled/Masks',
                                             # <- Change this per experiment
                                             label_path='./labels/FewShot/FFGEI_labels.csv')  # <- Change this for Graphcuts

    contingency_table = compare_evaluation([model1, model2], testing[0])
    print("contingency table::: ", contingency_table)
    statistics = mcnemars_statistic(contingency_table)
    return statistics

def mcnemars_statistic(table):
    # calculate mcnemar test
    result = mcnemar(table, exact=True)
    # summarize the finding
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')

    return [result.statistic, result.pvalue]


def extract_ttest_metrics(first_path, second_path, result_out):
    #Extract data
    sets = []
    sets.append(pd.read_csv(first_path))
    sets.append(pd.read_csv(second_path))

    #Get last 4 into new array
    pruned_sets = []
    for set in sets:
        listed = set.values.tolist()
        print("listed shape: ", len(listed), len(listed[0]))


        matrix_values = []
        for i, column in enumerate(zip(*listed)):
            print ("len: ", len(listed[0]))
            if i > len(listed[0]) - 5:
                print("in here: ", i)
                matrix_values.append(list(column))

        print("matrix values:")
        print(matrix_values)
        #Extract test results
        #listed = pruned.values.tolist()

        #Going to need to remove means and STD prior
        #Remove titles
        for i, column in enumerate(zip(*matrix_values)):
            print("value: ", column)
            for c in column:
                print("c is: ", c)
                for j, row in enumerate(matrix_values):
                    if c in row:
                        print("i found it", matrix_values[j], type(matrix_values[j]))
                        matrix_values[j].remove(c)
                    else:
                        print ("i cant find it")
            break

        print("pruned without column titles: ")
        print(matrix_values)

        pruned_sets.append(copy.deepcopy(matrix_values))

    #Translate into 4 separate 1 column arrays
    matrices = []
    print(type(pruned_sets), "is pruned set type")
    for i, set in enumerate(pruned_sets):
        for j, row in enumerate(set):
            for k, value in enumerate(row):
                pruned_sets[i][j][k] = float(pruned_sets[i][j][k])


    for iterator, set in enumerate(pruned_sets[0]):
        #Calculate TTest on all 4
        #make contingency table of set and pruned_sets[1][iterator]
        #print("set looks like: ")
        #print(set)
        #print("pruned looks like: ")
        #print(pruned_sets[1][iterator])
        #matrices.append(mcnemars_statistic(contingency_table))
        matrices.append(stats.ttest_ind(set, pruned_sets[1][iterator]))

    #Reshape into matrix
    #0, 1, 3, 2
    reshaped = [[matrices[0], matrices[1]], [matrices[3], matrices[2]]]

    #Print and return
    print("Confusion matrix of t-values:")
    print(reshaped)

    frame = pd.DataFrame([reshaped])
    print(frame.head())
    # Save as CSV all results
    os.makedirs('./Results/T_tests/', exist_ok=True)
    frame.to_csv('./Results/T_tests/' + result_out +  "_results.csv")

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def get_tiles(image_array):
    M = 80
    N = 80
    tiles = [image_array[x:x + M, y:y + N] for x in range(0, image_array.shape[0], M) for y in range(0, image_array.shape[1], N)]
    #Debug
    #for t in tiles:
    #    cv2.imshow("pasted ", np.asarray(t))
    #    key = cv2.waitKey(0) & 0xff
    return tiles

# Program to find most frequent element in a list
def most_frequent(List):
    return max(set(List), key=List.count)

def get_bounding_box(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y:y + h, x:x + w]
        cv2.imwrite(str(idx) + '.jpg', roi)
        cv2.rectangle(image,(x,y),(x+w,y+h),(255) ,2)
    return [x, y, w, h]

def get_bounding_mask(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    return contours


#Unravel FFGEI archive folders
def unravel_FFGEI(path = './Images/FFGEI/Unravelled/Mask'):
    #Rename all images according to its place in the queue (append with "z" to prevent duplicates)
    global_iter = 0
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        if len(files) > 0:
            images = []
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                #Stage one, rename all files numerically on a global scale
                os.rename(os.path.join(subdir, file), os.path.join(subdir, str(global_iter) + "z.jpg"))
                global_iter += 1
    #Second pass: extract all images into the same parent folder, removing the instance folders
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        if len(files) > 0:
            images = []
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                #Step two, now all files have unique names, move them to unravelled folder
                p = Path(os.path.join(subdir, file)).absolute()
                parent_dir = p.parents[1]
                p.rename(parent_dir / p.name)
                global_iter += 1

    #Third pass, remove the "z" from the files, this could potentially be compressed into the second pass.
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        if len(files) > 0:
            images = []
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                os.rename(os.path.join(subdir, file), os.path.join(subdir, file).replace('z', ''))

#Generate labels from processed images {0: chris, 1: claire}, chris is first in the fewShot dataset
def generate_labels(path, out, name, cutoff = 19, cutoff_index = 1):
    data = [['ID', 'Class']]
    global_iter = 0
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        print("directory: ", iterator, subdir, dir)
        if len(files) > 0:
            #Claire is index 1 - 20, Chris is the rest
            if cutoff_index == 1:
                index = 0
            else:
                index = 1

            if iterator > cutoff:
                index = cutoff_index
            images = []
            print("in here, ", subdir)
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                data.append([global_iter, index])
                global_iter += 1
        else:
            print("directory empty, iterating")
            
    os.makedirs(out, exist_ok=True)
    np.savetxt(out + name, data, delimiter=",", fmt='%s')
    print("end global iter: ", global_iter)

#Remove backgrounds in raw images to cut down on processing time
def remove_background_images(path):
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        print("directory: ", iterator, subdir)
        if len(files) > 0:
            images = []
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                # Get humans
                image = cv2.imread(os.path.join(subdir, file))
                objs = JetsonYolo.get_objs_from_frame(np.asarray(image), False)
                # If no humans detected, remove the image
                if len(objs) == 0:
                    os.remove(os.path.join(subdir, file))

#Remove all black images where no human has been found:
def remove_block_images(path):
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        if len(files) > 0:
            images = []
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                image = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)
                if np.all((image == 0)) or np.all((image == 255)):
                    os.remove(os.path.join(subdir, file))
                    
#Create directories if not already present
def make_directory(dir, text = "Couldn't make directory" ):
    os.makedirs(dir, exist_ok=True)

#For standard instances of extracting an array of images from an array of folders
def get_from_directory(path, exclusion = set(['FewShot'])):
    instances = []
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        if exclusion:
            dirs[:] = [d for d in dirs if d not in exclusion]
        dirs.sort(key = numericalSort)
        if len(files) > 0:
            images = []
            for file_iter, file in enumerate(sorted(files, key = numericalSort)):
                image = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)
                images.append(image)
            instances.append(images)
    return instances

#For standard instances of saving to directories
def save_to_directory(instances, path):
    os.makedirs(path, exist_ok=True)
    for instance in instances:
        #Find the latest un-made path and save the new images to it
        path_created = False
        n = 0.0
        while path_created == False:
            try:
                print("making path")
                local_path = path + "/Instance_" + str(n) + "/"
                os.mkdir(local_path)
                path_created = True
            except:
                print("invalid")
                n += 1
        for i, image in enumerate(instance):
            cv2.imwrite(local_path + str(i) + ".jpg", image)

#Align person in image
def align_image(image, thresh_area, thresh_width = 60, thresh_height = 100, HOG = False):
    processed_image = Image.new("L", (240,240))
    contours, heirarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #Return a black image for discarding unless contours have been detected or it is a HOG image
    if len(contours) > 0 or HOG == True:
        #Remove small contours indicative of noise
        threshold_cntrs = []
        for contour in contours:
            if cv2.contourArea(contour) > thresh_area:
                threshold_cntrs.append(contour)

        #Merge the large contours together
        if len(threshold_cntrs) > 0:
            merged_cntrs = threshold_cntrs[0]
            for i, c in enumerate(threshold_cntrs):
                if i != 0:
                    merged_cntrs = np.vstack([merged_cntrs, c])
            # Get and draw the bounding box of the merged contours
            x, y, w, h = cv2.boundingRect(merged_cntrs)
            #Restrict width
            if w > thresh_width:
                width_gap = w - thresh_width
                x = x + int(width_gap / 2)
                w = thresh_width
            # Extract, resize and centre the silhouette
            chopped_image = image[y:y + h, x:x + w]
            processed_image.paste(Image.fromarray(chopped_image), (120 - int(chopped_image.shape[1] / 2), y))
            #Debug
            #cv2.imshow("pasted ", np.asarray(processed_image))
            #key = cv2.waitKey(0) & 0xff
    return np.array(processed_image)

#################################################
#################################################