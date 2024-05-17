'''
This file contains all the methods related to the semi and unsupervised clustering experiments
'''
#libraries
import numpy as np
import copy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.stats import mode
#dependencies
from Programs.Data_Processing.kmeans_interp.kmeans_feature_imp import KMeansInterp
import Programs.Data_Processing.Utilities as Utilities


from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean

def calculate_centroid_distance(cluster_a_centroid, cluster_b_centroid, dimension_x):
    # Extract the centroids of cluster_a and cluster_b
    centroid_a = cluster_a_centroid[dimension_x]
    centroid_b = cluster_b_centroid[dimension_x]
    print("centroid distances: ", centroid_a, centroid_b)
    # Calculate the absolute distance between the centroids along dimension_x
    distance = np.abs(centroid_a - centroid_b)

    return distance

def calculate_distance_ranges(cluster_data, first_cluster_data, dimension_x):
    # Select data points belonging to cluster_a and cluster_b
    cluster_a_points = cluster_data.values.tolist()# cluster_data[cluster_labels == cluster_a]
    first_cluster_points = first_cluster_data.values.tolist()
    # Compute distances from cluster_a to cluster_b and vice versa along dimension_x
    print("CLUSTER DATA RANGE: ", cluster_data.shape)
    cluster_a_points = cluster_data.iloc[:, dimension_x].values  # Assuming x is the index, use .values to get the numpy array
    first_cluster_points = first_cluster_data.iloc[:, dimension_x].values  # Assuming x is the index, use .values to get the numpy array

    # Compute pairwise distances
    distances = np.abs(cluster_a_points[:, np.newaxis] - first_cluster_points)

 
    return distances.min(), distances.max()
from sklearn.metrics import adjusted_rand_score
from scipy.spatial import distance
def calculate_overlap_percentage(cluster_data, first_cluster_data, centroid_a, centroid_b, dimension):
    # Extract the specified dimension from each dataframe
    df1_dimension = cluster_data.iloc[:, dimension].values
    df2_dimension = first_cluster_data.iloc[:, dimension].values
    
    # Calculate distances from each point to both centroids for dataframe A
    distances_to_a = np.abs(df1_dimension - centroid_a[dimension])
    distances_to_b = np.abs(df1_dimension - centroid_b[dimension])
    
    # Count points closer to centroid B than centroid A in dataframe A
    count_a_to_b = np.sum(distances_to_b < distances_to_a)
    
    # Calculate distances from each point to both centroids for dataframe B
    distances_to_a = np.abs(df2_dimension - centroid_a[dimension])
    distances_to_b = np.abs(df2_dimension - centroid_b[dimension])
    
    # Count points closer to centroid B than centroid A in dataframe B
    count_b_to_b = np.sum(distances_to_b < distances_to_a)
    
    # Calculate intersection count
    intersection_count = min(count_a_to_b, count_b_to_b)
    
    # Calculate total number of points across both dataframes
    total_points = len(df1_dimension) + len(df2_dimension)
    
    # Calculate intersection ratio
    intersection_ratio = intersection_count / total_points
    
    return intersection_ratio

from itertools import combinations
def max_distance_between_centroids(centroids):
    max_distance = 0
    # Generate combinations of centroids
    centroid_combinations = combinations(centroids, 2)
    # Calculate distance between each pair of centroids
    for centroid_pair in centroid_combinations:
        distance = np.linalg.norm(centroid_pair[0] - centroid_pair[1])
        if distance > max_distance:
            max_distance = distance
    return max_distance

def k_means_experiment(data):
    print("In here??")
    data = pd.DataFrame(data)
    # Remove metadata columns and keep only the features and class
    features = data.iloc[:, 6:].values
    labels = data.iloc[:, 2].values 

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Standardize the features
    # Separate the data into labeled and unlabeled based on the class label
    '''
    labeled_indices = np.where(labels == 0)[0]
    unlabeled_indices = np.where(labels != 0)[0]

    labeled_data = features[labeled_indices]
    unlabeled_data = features[unlabeled_indices]
    labeled_data  = labeled_data.tolist()
    '''
    features = apply_grouped_pca(pd.DataFrame(features))
    pca_data_table = features.copy()
    pca_data_table['labels'] = labels

    print("feature shape: ", features.shape)

    # Initialize and fit KMeans model on the training data
    kmeans = KMeans(n_clusters=3)  # Assuming 3 clusters for the three classes
    kmeans.fit(features)
    cluster_labels = kmeans.labels_

    # Map clusters to the true class labels
    # For each cluster, find the most common true label
    cluster_to_class = {}
    for i in range(3):
        mask = cluster_labels == i
        cluster_to_class[i] = int(mode(labels[mask]).mode[0])
        

    # Map the cluster labels to the true class labels
    predicted_labels = np.array([cluster_to_class[cluster] for cluster in cluster_labels])
    accuracy = accuracy_score(labels, predicted_labels)

    #Re-arrange the cluster centroids to correspond to the classes 
    centroids = kmeans.cluster_centers_
    arranged_centroids = [None] * len(centroids)
    # Rearrange the sublists according to the dictionary
    for original_index, new_index in cluster_to_class.items():
        arranged_centroids[new_index] = centroids[original_index]

    distance_01 = np.linalg.norm(centroids[0] - centroids[1])  # Distance between cluster 0 and 1
    distance_02 = np.linalg.norm(centroids[0] - centroids[2])  # Distance between cluster 0 and 2
    distance_12 = np.linalg.norm(centroids[1] - centroids[2])  # Distance between cluster 1 and 2

    # Step 3: Calculate feature variability within each cluster
    cluster_variances = []
    cluster_datasets = []
    for cluster in range(3):
        #TODO: replace data labels in data with their predictions
        cluster_data = pca_data_table[pca_data_table.iloc[:, -1] == cluster]
        print("what's this: ", type(cluster_data), cluster_data.shape)
        cluster_datasets.append(cluster_data.iloc[:, :-1])
        cluster_variance = cluster_data.var()
        #remove noise
        #if cluster_variance > 1000:
        #    cluster_variance = 0.0

        cluster_variances.append(cluster_variance)
        

    print("cluster variances: ", type(cluster_variances[0]), cluster_variances[0][0])


    # Combine the feature differences and variabilities to rank features for each cluster
    # Create a dictionary to store the importance scores
    #feature_importance = {}
    all_importances  = []
    max_dist = max_distance_between_centroids(centroids)
    for feature_index in range(18):
        print("FEATURE INDEX: ", feature_index)
        importance_scores = []
        for cluster in range(1,3):
            # calculate each abnormal cluster with initial cluster
            #f=σf2​1​(1+similarityf,α,β​)+max(⋅)−min(⋅)∣μα​−μβ​∣​

            '''
            Feature importance f = 1/cluster_variance_f  * (1 + cluster_overlap for feature f between cluster a and b)
              + (abs(centroid_a - centroid_b) / the maximum feature value across clusters a and b - the minimum feature value across clusters a and b)
            '''
            print("variance: ", cluster, feature_index, cluster_variances[cluster][feature_index])
            w1 = 1.0
            w2 = 0.5
            w3 = 1.0
            cluster_variance_inv = 1/cluster_variances[cluster][feature_index]
            cluster_variance_inv *= w1
            cluster_overlap_f = calculate_overlap_percentage(cluster_datasets[cluster], cluster_datasets[0], centroids[cluster], centroids[0], dimension=feature_index)
            #cluster_overlap_f += 1
            cluster_overlap_f *= w2
            centroid_dist_f = calculate_centroid_distance(centroids[cluster], centroids[0], feature_index) 
            norm_centroid_dist_f = centroid_dist_f# / max_dist
            norm_centroid_dist_f *= w3
            feature_importance_f = (cluster_variance_inv * cluster_overlap_f) * norm_centroid_dist_f
            # Example usage:

            print("Overlap percentages along dimension x for each cluster:", cluster_variance_inv, cluster_overlap_f, centroid_dist_f)
            print("final feature importance: ", feature_importance_f)
            if pd.isna(feature_importance_f):
                feature_importance_f = 0.0
            #importance = 1 / ((cluster_variances[cluster][feature_index] * centroid_differences[feature_index]) + 0.0001)

            importance_scores.append([cluster, feature_importance_f])
        all_importances.append(importance_scores)
    for imp in all_importances:
        print("final importance scores: ", imp)

    limp_importance = [row[0] for row in all_importances]
    shuffle_importance = [row[1] for row in all_importances]

    #features.columns = ['Nose','L_eye','R_eye','L_ear','R_ear','L_shoulder','R_shoulder',
    #'L_elbow','R_elbow','L_hand','R_hand','L_hip','R_hip','L_knee','R_knee','L_foot', 'R_foot', 'M_hip']
    limp_body_importance = {'head':0, 'torso': 0, 'left_arm': 0, 'right_arm': 0, 'left_leg': 0, 'right_leg': 0}
    for ind, importance in enumerate(limp_importance):
        print("ind and importance: ", ind, importance)
        if ind in [0,1,2,3,4]: #1,2,3,4
            limp_body_importance['head'] += importance[1] * 0.4
        elif ind in [3,6,11,17,12]: #0,17
            limp_body_importance['torso'] += importance[1] * 0.4
        elif ind in [7,9]:
            limp_body_importance['left_arm'] += importance[1]
        elif ind in [8,10]:
            limp_body_importance['right_arm'] += importance[1]
        elif ind in [13,15]:
            limp_body_importance['left_leg'] += importance[1]
        elif ind in [14,16]:
            limp_body_importance['right_leg'] += importance[1]
        else:
            print("something gone wrong")


    shuffle_body_importance = {'head':0, 'torso': 0, 'left_arm': 0, 'right_arm': 0, 'left_leg': 0, 'right_leg': 0}
    for ind, importance in enumerate(shuffle_importance):
        print("ind and importance: ", ind, importance)
        if ind in [1,2,3,4]:
            shuffle_body_importance['head'] += importance[1]
        elif ind in [0, 17]:
            shuffle_body_importance['torso'] += importance[1]
        elif ind in [5,7,9]:
            shuffle_body_importance['left_arm'] += importance[1]
        elif ind in [6,8,10]:
            shuffle_body_importance['right_arm'] += importance[1]
        elif ind in [11,13,15]:
            shuffle_body_importance['left_leg'] += importance[1]
        elif ind in [12,14,16]:
            shuffle_body_importance['right_leg'] += importance[1]
        else:
            print("something gone wrong")

    total_sum = sum(limp_body_importance.values())
    # Replace each value with its percentage of the total sum
    percentage_limp = {k: (v / total_sum) * 100 for k, v in limp_body_importance.items()}
    total_sum = sum(limp_body_importance.values())
    # Replace each value with its percentage of the total sum
    percentage_shuffle = {k: (v / total_sum) * 100 for k, v in shuffle_body_importance.items()}

    percentage_limp = sorted(percentage_limp.items(), key=lambda item: item[1])
    percentage_shuffle = sorted(percentage_shuffle.items(), key=lambda item: item[1])
    print(percentage_limp)
    print(percentage_shuffle)
    (l_lk, l_lv) = percentage_limp[-1]
    (l_sk, l_sv) = percentage_shuffle[-1]
    print("what are these: ", (l_lk, l_lv) , "separeate", (l_sk, l_sv) )
    #Need to do PCA on original features 
    features.columns = ['Nose','L_eye','R_eye','L_ear','R_ear','L_shoulder','R_shoulder',
    'L_elbow','R_elbow','L_hand','R_hand','L_hip','R_hip','L_knee','R_knee','L_foot', 'R_foot', 'M_hip']

    # Combine Joints into joint-groups
    features['head'] = features[['L_eye', 'R_eye', 'L_ear', 'R_ear']].sum(axis=1)
    features['torso'] = features[['Nose', 'M_hip']].sum(axis=1)
    features['left_arm'] = features[['L_shoulder', 'L_elbow', 'L_hand']].sum(axis=1)
    features['right_arm'] = features[['R_shoulder', 'R_elbow', 'R_hand']].sum(axis=1)
    features['left_leg'] = features[['L_hip', 'L_knee', 'L_foot']].sum(axis=1)
    features['right_leg'] = features[['R_hip', 'R_knee', 'R_foot']].sum(axis=1)
    # Create a new DataFrame with the combined columns
    X = features[['head', 'left_arm', 'right_arm', 'left_leg', 'right_leg', 'torso']]
    kms = KMeansInterp(
        n_clusters=3,
        ordered_feature_names=X.columns.tolist(), 
        feature_importance_method='wcss_min', # or 'unsup2sup'
    ).fit(X.values)

    # A dictionary where the key [0] is the cluster label, and [:10] will refer to the first 10 most important features
    print("cluster 0", kms.feature_importances_[0][:10])# Features here are words
    print("cluster 1", kms.feature_importances_[1][:10])# Features here are words
    print("cluster 2", kms.feature_importances_[2][:10])# Features here are words
    print("Accuracy", accuracy)
    return [kms.feature_importances_[0][0], percentage_limp, percentage_shuffle], [distance_01, distance_02, distance_12], kmeans, cluster_to_class


def apply_grouped_pca(data):
   # Initialize a list to store the PCA features for each group
    pca_features = []

    # Number of groups
    num_groups = 18

    # Apply PCA to each group of 6 features
    for group_num in range(num_groups):
        # Calculate the indices for the current group
        feature_indices = [group_num + i * num_groups for i in range(6)]
        
        # Select the group of 6 features
        feature_group = data.iloc[:, feature_indices]
        
        # Initialize PCA
        pca = PCA(n_components=1)  # Assuming you want to keep 1 principal component per group
        
        # Fit PCA to the feature group and transform it
        pca_result = pca.fit_transform(feature_group)
        
        # Append the PCA feature to the list
        pca_features.append(pca_result.flatten())

    # Concatenate the PCA features into a DataFrame
    pca_df = pd.DataFrame(pca_features).T

    # Rename the columns to represent the PCA features
    pca_df.columns = [f'PCA_Feature_{group_num + 1}' for group_num in range(num_groups)]

    # Display the resulting DataFrame
    return pca_df


def apply_standard_scaler(data, output):
    data, _ = Utilities.process_data_input(data, None)
    #remove all metadata
    meta_data = [row[:6] for row in data]
    joints_data = [row[6:] for row in data]
    print("correct?, ", meta_data[0])
    print("and this: ", joints_data[0])
    #unwrap all joints
    unwrapped_joints = [[value for sublist in row for value in sublist] for row in joints_data]
    print("unwrap: ", unwrapped_joints[0], len(unwrapped_joints[0]))
    #apply scaler
    # Initialize StandardScaler
    scaler =  MinMaxScaler(feature_range=(0, 1))

    # Fit the scaler to your data (calculate mean and standard deviation)
    scaler.fit(unwrapped_joints)

    # Transform the data (apply standard scaling)
    scaled_joints = scaler.transform(unwrapped_joints)
    #rewrap all joints
    rewrapped_joints = []
    for i, row in enumerate(scaled_joints):
        coord = []
        joints_row = []
        for j, val in enumerate(row):
            if j != 0 and j % 3 == 0:
                joints_row.append(copy.deepcopy(coord))
                coord = []
            coord.append(val)
        joints_row.append(copy.deepcopy(coord))
        rewrapped_joints.append(joints_row)

    #print("rewrapped: ", rewrapped_joints[0], len(rewrapped_joints[0]))
    #stop = 5/0
    #readd metadata
    for i, row in enumerate(rewrapped_joints):
        #print("prior:", rewrapped_joints[i])
        #print("sizes? ", len(rewrapped_joints[i]))
        rewrapped_joints[i][:0] = meta_data[i]
        #print("readded:", rewrapped_joints[i])
        #print("sizes? ", len(rewrapped_joints[i]))
        #stop = 5/0

    Utilities.save_dataset(rewrapped_joints, output)

def stitch_data_for_kmeans(data):
    new_data = []
    new_row = []
    counter = 0
    for i, row in enumerate(data):
        #print("length after row: ", i , len(new_row))
        if counter == 6 and i != 0:
            #print("counter called at :", i, len(new_row))
            new_data.append(copy.deepcopy(new_row))
            new_row = []
            counter = 0

        #print("row len: ", i, len(row))
        for j, val in enumerate(row):
            if len(new_row) < 6:
                new_row.append(val)
            elif j > 5: 
                new_row.append(val)

        
        counter += 1
    
    print("new row: ", len(new_data), len(new_data[0]), len(new_data[1]), len(new_data[-1]))
    print("row 0: ", new_data[0])
    print("row 1: ", new_data[1])
    print("last row: ", new_data[-1])
    return new_data

import random

def remove_incorrect_predictions(data):
    new_data = []
    print("original : ", len(data))
    for i, row in enumerate(data):
        if row[2] == 0 and data[i][6] == 1:
            new_data.append(row)
        elif row[2] == 1 and data[i][7] == 1:
            new_data.append(row)
        elif row[2] == 2 and data[i][8] == 1:
            new_data.append(row)
    print("final: ", len(new_data))
    return new_data

def fix_incorrect_data(data):
    for i, row in enumerate(data):
        if row[2] == 0:
            data[i][6] = random.uniform(0.7, 1.0)
            data[i][7] = random.uniform(0.2, 0.6)
            data[i][8] = random.uniform(0.0, 0.8)
        elif row[2] == 1:
            data[i][6] = random.uniform(0.2, 0.6)
            data[i][7] = random.uniform(0.7, 1.0)
            data[i][8] = random.uniform(0.0, 0.8)
        elif row[2] == 2:  
            data[i][6] = random.uniform(0.0, 0.8)
            data[i][7] = random.uniform(0.2, 0.6)
            data[i][8] = random.uniform(0.7, 1.0)
    return data

def calculate_column_averages(data):
    # Initialize sums for each column
    col_sums = [0, 0, 0]

    # Iterate through each sublist (row)
    for row in data:
        # Accumulate the sum for each column
        col_sums[0] += row[0]
        col_sums[1] += row[1]
        col_sums[2] += row[2]

    # Calculate the average for each column
    num_rows = len(data)
    col_averages = [col_sums[i] / num_rows for i in range(3)]

    return col_averages

def map_predictions(predictions, cluster_map):
    """
    Map the cluster predictions according to the provided cluster mapping.

    Parameters:
        predictions (list or numpy array): Original cluster predictions from k-means.
        cluster_map (dict): Dictionary mapping original clusters to new clusters.
        
    Returns:
        list or numpy array: Mapped cluster predictions.
    """
    # Apply the cluster mapping to each prediction
    mapped_predictions = [cluster_map[pred] for pred in predictions]
    
    return mapped_predictions

def predict_and_calculate_proximity(kmeans_model, data_df, metadata, cluster_map):
    """
    Predicts the cluster each data instance belongs to and calculates the proximity (distance)
    to each of the k-means model's centroids.
    
    Parameters:
    kmeans_model (KMeans): The trained KMeans model.
    data_df (DataFrame): The DataFrame containing data instances.

    Returns:
    DataFrame: A DataFrame with the predictions and proximity values.
    """
    # Convert the DataFrame to a NumPy array for efficient calculation
    data_array = data_df.to_numpy()
    
    # Predict the clusters for each instance using the KMeans model
    cluster_predictions = kmeans_model.predict(data_array)

    cluster_predictions = map_predictions(cluster_predictions, cluster_map)
    #cluster_accuracy = accuracy_score(labels, cluster_predictions)
    ##print("accuracy in here: ", cluster_accuracy)
    
    # Get the centroids from the KMeans model
    centroids = kmeans_model.cluster_centers_
    
    # List to store the proximity values for each instance
    proximities = []

    # Calculate proximity to each centroid for each data instance
    for instance in data_array:
        # Calculate distances to each centroid
        distances = [np.linalg.norm(instance - centroid) for centroid in centroids]
        
        # Append the list of distances to the proximities list
        proximities.append(distances)
    
    # Create a new DataFrame with predictions and proximities
    result_df = metadata
    for i, v in enumerate(proximities):
        proximities[i] = gait_coefficient(v[0], v[1], v[2])
    result_df['Cluster'] = cluster_predictions
    result_df['Severity coefficient'] = proximities
    calculate_mean_variance(result_df['Cluster'], result_df['Severity coefficient'], cluster_map)
    return result_df.values.tolist()

def gait_coefficient(d0, d1, d2, w1 = 0.5, w2 = 1.5):
    """
    Calculate the coefficient representing how far an individual's gait pattern is from regular gait.

    Parameters:
        d0 (float): Distance from the individual's gait pattern to the centroid of cluster 0 (regular gait).
        d1 (float): Distance from the individual's gait pattern to the centroid of cluster 1 (first type of pathology).
        d2 (float): Distance from the individual's gait pattern to the centroid of cluster 2 (more severe type of pathology).
        w1 (float): Weight for the distance to cluster 1 (default is 0.5).
        w2 (float): Weight for the distance to cluster 2 (default is 1.0).

    Returns:
        float: Coefficient representing how far the individual's gait pattern is from regular gait.
    """
    # Calculate the weighted distances to clusters 1 and 2 relative to the distance to cluster 0
    weighted_d1 = w1 * (d1 / d0)
    weighted_d2 = w2 * (d2 / d0)
    
    # Calculate the total coefficient as the sum of weighted distances
    coefficient = weighted_d1 + weighted_d2
    
    return coefficient

def calculate_mean_variance(labels, coefficients, cluster_map):
    """
    Calculate and print the mean and variance of coefficients for each unique label.

    Parameters:
        labels (pandas Series): Series containing label values.
        coefficients (pandas Series): Series containing coefficient values.
    """
    # Combine the labels and coefficients into a DataFrame
    data = pd.DataFrame({'label': labels, 'coefficient': coefficients})

    # Group by labels
    grouped_data = data.groupby('label')

    # Calculate mean and variance for each group
    mean_vars = [0 for i in grouped_data]
    for label, group in grouped_data:
        mean = group['coefficient'].mean()
        variance = group['coefficient'].var()
        mean_vars[int(cluster_map[label])] = [mean, variance]
        print("applying: ", mean, " to : ", cluster_map, cluster_map[label], label)
    for i, (mean, var) in enumerate(mean_vars):
        print(f"Label {i}: Mean = {mean:.4f}, Variance = {var:.4f}")

def unsupervised_cluster_assessment(input, output, epochs = 15):
    print("starting")

    data, _ = Utilities.process_data_input(input, None)
    data = stitch_data_for_kmeans(data)
    Utilities.save_dataset(data, './code/datasets/joint_data/embed_data/2_people_fixed')

    feature_counts ={0: {'head': 0, 'left_arm': 0, 'right_arm': 0, 'left_leg': 0, 'right_leg': 0, 'torso': 0},
                     1: {'head': 0, 'left_arm': 0, 'right_arm': 0, 'left_leg': 0, 'right_leg': 0, 'torso': 0},
                     2: {'head': 0, 'left_arm': 0, 'right_arm': 0, 'left_leg': 0, 'right_leg': 0, 'torso': 0}
                     }
    
    centroid_distances = []
    for i in range(epochs):
        [clust_1, clust_2, clust_3], [dist_01, dist_02, dist_12], k_model, cluster_map = k_means_experiment(data)
        print("what are these: ", clust_1, clust_2, clust_3, "space: ", clust_2[0], clust_2[0][0], clust_2[0][1], type(clust_2))
        #stop = 5/0
        for j in range(len(clust_2)):
            feature_counts[1][clust_2[j][0]] += clust_2[j][1]/epochs
        for j in range(len(clust_3)):
            feature_counts[2][clust_3[j][0]] += clust_3[j][1]/epochs
        feature_counts[0][clust_1[0]] += 1
        #feature_counts[1][clust_2[0]] += 1
        #feature_counts[2][clust_3[0]] += 1
        centroid_distances.append([dist_01, dist_02, dist_12])


    averages = calculate_column_averages(centroid_distances)
    feature_counts = {
        outer_k: {inner_k: round(inner_v, 2) for inner_k, inner_v in outer_v.items()}
        for outer_k, outer_v in feature_counts.items()
    }
    feature_counts = {
        k: {ik: iv for ik, iv in sorted(v.items(), key=lambda item: item[1], reverse=True)}
        for k, v in feature_counts.items()
    }

    print("feature counts: ", feature_counts)
    print("centroid distances: ", averages)

    #process data for k-means experiments
    data  = pd.DataFrame(data)
    features = data.iloc[:, 6:].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = apply_grouped_pca(pd.DataFrame(features))
    result_df = predict_and_calculate_proximity(k_model, features, data.iloc[:, :6], cluster_map)
    print("cluster map", cluster_map)
    #Add column names
    Utilities.save_dataset(result_df, output)
    return result_df, k_model, cluster_map, feature_counts