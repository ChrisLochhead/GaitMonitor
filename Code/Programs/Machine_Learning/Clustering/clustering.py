'''
This file contains all the methods related to the semi and unsupervised clustering experiments
'''
#libraries
import numpy as np
import copy
import os
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import mode
from itertools import combinations
#dependencies
from Programs.Data_Processing.kmeans_interp.kmeans_feature_imp import KMeansInterp
import Programs.Data_Processing.Utilities as Utilities

def calculate_centroid_distance(cluster_a_centroid, cluster_b_centroid, dimension_x):
    '''
    Calculates the difference between the centroid of the normal cluster from all other clusters along a specific feature dimension.

    Arguments
    ---------
    cluster_a_centroid: list(float)
        centroid of the "normal" cluster
    cluster_b_centroid: list(float)
        centroid of a different cluster
    dimension_x: int
        index of the feature to calculate the distance between

    Returns
    -------
    Float
        Returns the euclidean distance between the two centroids on dimension x

    '''
    # Extract the centroids of cluster_a and cluster_b
    centroid_a = cluster_a_centroid[dimension_x]
    centroid_b = cluster_b_centroid[dimension_x]
    print("centroid distances: ", centroid_a, centroid_b)
    # Calculate the absolute distance between the centroids along dimension_x
    distance = np.abs(centroid_a - centroid_b)
    return distance

def calculate_distance_ranges(cluster_data, first_cluster_data, dimension_x):
    '''
    Calculates the maximum and minimum distances of instances in two clusters along a specified dimension

    Arguments
    ---------
    cluster_data: Pandas.Dataframe
        dataframe for all instances in a cluster
    first_cluster_data: Pandas.Dataframe
        dataframe for all instances in the "normal" cluster
    dimension_x: int
        index of the feature to calculate the distances between

    Returns
    -------
    Float, Float
        Returns the maximum and minimum euclidean distances between the two clusters on dimension x

    '''
    # Select data points belonging to cluster_a and cluster_b
    cluster_a_points = cluster_data.values.tolist()
    first_cluster_points = first_cluster_data.values.tolist()
    # Compute distances from cluster_a to cluster_b and vice versa along dimension_x
    cluster_a_points = cluster_data.iloc[:, dimension_x].values  # Assuming x is the index, use .values to get the numpy array
    first_cluster_points = first_cluster_data.iloc[:, dimension_x].values  # Assuming x is the index, use .values to get the numpy array
    # Compute pairwise distances
    distances = np.abs(cluster_a_points[:, np.newaxis] - first_cluster_points)
    return distances.min(), distances.max()

def calculate_overlap_percentage(cluster_data, first_cluster_data, centroid_a, centroid_b, dimension):
    '''
    Calculates the overlap between two clusters along a specified dimension

    Arguments
    ---------
    cluster_data: Pandas.Dataframe
        dataframe for all instances in a cluster
    first_cluster_data: Pandas.Dataframe
        dataframe for all instances in the "normal" cluster
    centroid_a: Pandas.Series
        centroid of cluster data
    centroid_b: Pandas.Series
        centroid of first_cluster_data
    dimension: int
        index of the feature to calculate the overlap between

    Returns
    -------
    Float
        Returns the intersection ratio between the two clusters on the specified dimensional plane

    '''
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
    return 1 - intersection_ratio

def k_means_experiment(data, num_classes = 3, normal_class = 0):
    '''
    Runs a single k-means model on the data to get the clustered predictions

    Arguments
    ---------
    data: List(List)
        input data in the correct format 
    num_classes: int (optional: default = 3)
        number of output classes to expect and to initialize the k-means model to.

    Returns
    -------
    List(int), List(float), K-means.model, dict(int:int)
        Returns the percentage importance of each feature, their distance values from the "normal" centroid, the trained model and a 
        dictionary indicating the mapping from the models predicted classes from 0-n to the actual order of those classes in the data.
    '''
    data = pd.DataFrame(data)
    # Remove metadata columns and keep only the features and class
    features = data.iloc[:, 6:].values
    labels = data.iloc[:, 2].values 

    # Standardize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    # Separate the data into labeled and unlabeled based on the class label
    features = apply_grouped_pca(pd.DataFrame(features))
    pca_data_table = features.copy()
    pca_data_table['labels'] = labels

    # Initialize and fit KMeans model on the training data
    kmeans = KMeans(n_clusters=num_classes)  # Assuming 3 clusters for the three classes
    kmeans.fit(features)
    cluster_labels = kmeans.labels_

    # Map clusters to the true class labels
    # For each cluster, find the most common true label
    cluster_to_class = {}
    for i in range(num_classes):
        mask = cluster_labels == i
        cluster_to_class[i] = int(mode(labels[mask]).mode[0])
        
    # Map the cluster labels to the true class labels
    predicted_labels = np.array([cluster_to_class[cluster] for cluster in cluster_labels])
    accuracy = accuracy_score(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels, average='weighted')
    #for i, val in enumerate(zip(labels, predicted_labels)):
    #    print("key value pair: ", val)
    #print("are these right: ", len(labels), len(predicted_labels))

    #Re-arrange the cluster centroids to correspond to the classes 
    centroids = kmeans.cluster_centers_
    arranged_centroids = [None] * len(centroids)
    # Rearrange the sublists according to the dictionary
    for original_index, new_index in cluster_to_class.items():
        arranged_centroids[new_index] = centroids[original_index]

    distances = [np.linalg.norm(centroids[0] - centroids[i]) for i in range(len(centroids)) ]
    # Step 3: Calculate feature variability within each cluster
    cluster_variances = []
    cluster_datasets = []
    for cluster in range(num_classes):
        #TODO: replace data labels in data with their predictions
        cluster_data = pca_data_table[pca_data_table.iloc[:, -1] == cluster]
        cluster_datasets.append(cluster_data.iloc[:, :-1])
        cluster_variance = cluster_data.var()
        cluster_variances.append(cluster_variance)
        
    all_importances  = []
    for feature_index in range(18):
        importance_scores = []
        for cluster in range(num_classes):
            '''
            Feature importance f = 1/cluster_variance_f  * (1 + cluster_overlap for feature f between cluster a and b)
              + (abs(centroid_a - centroid_b) / the maximum feature value across clusters a and b - the minimum feature value across clusters a and b)
            '''
            w1 = 1.0
            w2 = 1.0
            w3 = 1.0
            cluster_variance_inv = 1/cluster_variances[cluster][feature_index]
            cluster_variance_inv *= w1
            cluster_overlap_f = calculate_overlap_percentage(cluster_datasets[cluster], cluster_datasets[normal_class], centroids[cluster], centroids[normal_class], dimension=feature_index)
            cluster_overlap_f *= w2
            centroid_dist_f = calculate_centroid_distance(centroids[cluster], centroids[normal_class], feature_index) 
            #max_dist = calculate_distance_ranges(pd.DataFrame(centroids[cluster]), pd.DataFrame(centroids[normal_class]), feature_index)
            norm_centroid_dist_f = centroid_dist_f#/ max_dist
            norm_centroid_dist_f *= w3
            print("wtaf are these: ", norm_centroid_dist_f, cluster_variance_inv, cluster_overlap_f)
            feature_importance_f = (cluster_variance_inv * cluster_overlap_f)  * (1 + norm_centroid_dist_f)

            # Example usage:

            if pd.isna(feature_importance_f):
                feature_importance_f = 0.0

            importance_scores.append([cluster, feature_importance_f])
        all_importances.append(importance_scores)

    class_importances = [[] for i in range(num_classes)]
    print("class lens:", len(class_importances))
    for i in range(len(all_importances)):
        for j in range(num_classes):
            print("£in here : ", j)
            print("class: ", class_importances[j])
            class_importances[j].append([row[j] for row in all_importances])

    print("only two right?", len(class_importances), class_importances[0], len(class_importances[0]))

    #features.columns = ['Nose','L_eye','R_eye','L_ear','R_ear','L_shoulder','R_shoulder',
    #'L_elbow','R_elbow','L_hand','R_hand','L_hip','R_hip','L_knee','R_knee','L_foot', 'R_foot', 'M_hip']
    region_importances = [{'head':0, 'torso': 0, 'left_arm': 0, 'right_arm': 0, 'left_leg': 0, 'right_leg': 0} for i in range(num_classes)]
    print("iterators: ", len(region_importances), len(class_importances))
    for i, region_importance in enumerate(region_importances):
            for ind, importance in enumerate(class_importances[i]):
                print("ind and importance: ", ind, importance)
                for ind_i, feature_importance in enumerate(importance):
                    #stop = 5/0
                    if ind_i in [1,2,3,4]: #1,2,3,4
                        region_importances[i]['head'] += feature_importance[1] / 2.5# * 0.4
                    elif ind_i in [0,17]: #0,17
                        region_importances[i]['torso'] += feature_importance[1] / 1.25# * 0.4
                    elif ind_i in [5, 7,9]:
                        region_importances[i]['left_arm'] += feature_importance[1] / 3
                    elif ind_i in [6, 8,10]:
                        region_importances[i]['right_arm'] += feature_importance[1] / 3
                    elif ind_i in [11,13,15]:
                        region_importances[i]['left_leg'] += feature_importance[1] /1.5
                    elif ind_i in [12,14,16]:
                        region_importances[i]['right_leg'] += feature_importance[1] /1.5
                    else:
                        print("something gone wrong")


    total_sums = [sum(region_importances[i].values()) for i in range(len(region_importances))]
    # Replace each value with its percentage of the total sum
    total_percentages = {k: (v / (total_sums[i] + 0.0001)) * 100 for k, v in region_importances[i].items() for i in range(len(region_importances))}
    total_percentages = [sorted(region_importances[i].items(), key=lambda item: item[1]) for i in range(len(region_importances))]

    #print("what's this: ", total_percentages)
    #stp = 5/0
    #features.columns = ['Nose','L_eye','R_eye','L_ear','R_ear','L_shoulder','R_shoulder',
    #'L_elbow','R_elbow','L_hand','R_hand','L_hip','R_hip','L_knee','R_knee','L_foot', 'R_foot', 'M_hip']

    # Combine Joints into joint-groups
    #features['head'] = features[['L_eye', 'R_eye', 'L_ear', 'R_ear']].sum(axis=1)
    #features['torso'] = features[['Nose', 'M_hip']].sum(axis=1)
    #features['left_arm'] = features[['L_shoulder', 'L_elbow', 'L_hand']].sum(axis=1)
    #features['right_arm'] = features[['R_shoulder', 'R_elbow', 'R_hand']].sum(axis=1)
    #features['left_leg'] = features[['L_hip', 'L_knee', 'L_foot']].sum(axis=1)
    #features['right_leg'] = features[['R_hip', 'R_knee', 'R_foot']].sum(axis=1)
    # Create a new DataFrame with the combined columns
    #X = features[['head', 'left_arm', 'right_arm', 'left_leg', 'right_leg', 'torso']]
    #kms = KMeansInterp(
    #    n_clusters=num_classes,
    #    ordered_feature_names=X.columns.tolist(), 
    #    feature_importance_method='wcss_min', # or 'unsup2sup'
    #).fit(X.values)

    # A dictionary where the key [0] is the cluster label, and [:10] will refer to the first 10 most important features
    print("Accuracy", accuracy, f1)
    #stop = 5/0
    return total_percentages, distances, kmeans, cluster_to_class


def apply_grouped_pca(data, num_groups = 18):
    '''
    Apply Principal Component Analysis to the grouped data, converting embedded 7-frame long pieces of data into a single frame embedding

    Arguments
    ---------
    data: List(List)
        input data in the correct format 
    num_groups: int (optional, default = 18)
        number of nodes to expect per-frame

    Returns
    -------
    Pandas.Dataframe
        A dataframe of the newly converted PCA data.
    '''
   # Initialize a list to store the PCA features for each group
    pca_features = []

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
        #print("shapes: ", len(pca_result), len(pca_result[0]), feature_group.shape, feature_group.iloc[:, 2].shape)
        pca_features.append(pca_result.flatten())
        #pca_features.append(feature_group.iloc[:, 1])
        #stop = 5/0

    # Concatenate the PCA features into a DataFrame
    pca_df = pd.DataFrame(pca_features).T
    # Rename the columns to represent the PCA features
    pca_df.columns = [f'PCA_Feature_{group_num + 1}' for group_num in range(num_groups)]
    return pca_df

def apply_standard_scaler(data, output):
    '''
    Apply a standard scaler to the data to assist in k-means convergence

    Arguments
    ---------
    data: List(List)
        input data in the correct format 
    output : str
        the file location for the outputted data

    Returns
    -------
    Pandas.Dataframe
        A dataframe of the newly converted PCA data.
    '''
    data, _ = Utilities.process_data_input(data, None)
    #remove all metadata
    meta_data = [row[:6] for row in data]
    joints_data = [row[6:] for row in data]
    #unwrap all joints
    unwrapped_joints = [[value for sublist in row for value in sublist] for row in joints_data]
    #apply scaler
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

    for i, row in enumerate(rewrapped_joints):
        rewrapped_joints[i][:0] = meta_data[i]
    Utilities.save_dataset(rewrapped_joints, output)

def stitch_data_for_kmeans(data):
    '''
    Stitches the per-frame dataset into sequences of 7 frames, essentially turning them into the same format as they are in the
    custom datasets based on pytorch datasets in this codebase.

    Arguments
    ---------
    data: List(List)
        input data in the correct format 

    Returns
    -------
    List(List)
        The data in the new sequential format.
    '''
    new_data = []
    new_row = []
    counter = 0
    for i, row in enumerate(data):
        if counter == 6 and i != 0:
            new_data.append(copy.deepcopy(new_row))
            new_row = []
            counter = 0

        for j, val in enumerate(row):
            if len(new_row) < 6:
                new_row.append(val)
            elif j > 5: 
                new_row.append(val)
        counter += 1
    return new_data

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

def predict_and_calculate_proximity(kmeans_model, data_df, metadata, cluster_map, feature_counts, normal_class):
    """
    Predicts the cluster each data instance belongs to and calculates the proximity (distance)
    to each of the k-means model's centroids.
    
    Parameters
    -----------------------
    kmeans_model (KMeans):
        The trained KMeans model.
    data_df (DataFrame):
        The DataFrame containing data instances.
    normal_class: int (optional, default = 0)
        Index in the data denoting "normal" gait. Usually but not always the first example

    Returns
    -----------------------
    DataFrame: A DataFrame with the predictions and proximity values.
    """
    data_array = data_df.to_numpy()
    # Predict the clusters for each instance using the KMeans model
    cluster_predictions = kmeans_model.predict(data_array)
    cluster_predictions = map_predictions(cluster_predictions, cluster_map)
    
    # Get the centroids from the KMeans model
    centroids = kmeans_model.cluster_centers_
    proximities = []
    # Calculate proximity to each centroid for each data instance
    for instance in data_array:
        # Calculate distances to each centroid
        distances = [np.linalg.norm(instance - centroid) for centroid in centroids]
        # Append the list of distances to the proximities list
        new_array = [None] * len(distances)
        for key, value in cluster_map.items():
            new_array[key] = distances[value]
        proximities.append(new_array)
       
    # Create a new DataFrame with predictions and proximities
    result_df = metadata
    for i, v in enumerate(proximities):
        proximities[i] = gait_coefficient(v, cluster_predictions[i], feature_importances=feature_counts, normal_class=normal_class)
        print("severity prediction: ", proximities[i], cluster_predictions[i], v)
    print("cluster map:", cluster_map)
    #stop = 5/0
    result_df['Cluster'] = cluster_predictions
    result_df['Severity coefficient'] = proximities
    calculate_mean_variance(result_df['Cluster'], result_df['Severity coefficient'], cluster_map)
    return result_df.values.tolist()

def gait_coefficient(distances, cluster_prediction, weights = [1.0, 1.0, 1.0], feature_importances = None, normal_class = 0):
    """
    Calculate the coefficient representing how far an individual's gait pattern is from regular gait.

    Parameters:
        distances: list(float)
            Distance from the individual's gait pattern to the centroid of cluster 0 (regular gait).
        cluster_prediction: list(int)
            Predictions for each example
        weights: list(float) (optional, default = [0.5, 1.5] )
            weigths for each pathology (experimental)
        normal_class: int (optional, default = 0)
            Index in the data denoting "normal" gait. Usually but not always the first example
    Returns:
        float: Coefficient representing how far the individual's gait pattern is from regular gait.
    """
    # Calculate the weighted distances to clusters 1 and 2 relative to the distance to cluster 0

    #this has to be imbued with context, k-means aims to maximize difference across all features, so features need to be weighted by importance to calculate the real severity co-efficient.
    #importance class 0 feature n - importance class 1 feature n to get co-efficient and multiply by distances
    #print("distances: ", distances, normal_class, cluster_prediction, distances[normal_class], distances[cluster_prediction])
    #stop = 5/0
    num_classes = len(distances)
    regularizer = 0
    for i in range(num_classes):
        if i not in [normal_class]:
            regularizer += (distances[i] / num_classes)

    return (distances[normal_class]) / regularizer 

def aggregate_distances(distance, n):
    distance_aggregate = [0 for i in range(6)]
    for ind, value in enumerate(distance):
        if ind in [1,2,3,4]: #1,2,3,4
            distance_aggregate[0] += value / 4
        elif ind in [0,17]: #0,17
            distance_aggregate[1] += value / 2
        elif ind in [5, 7,9]:
            distance_aggregate[2] += value / 3
        elif ind in [6, 8,10]:
            distance_aggregate[3] += value / 3
        elif ind in [11,13,15]:
            distance_aggregate[4] += value / 3
        elif ind in [12,14,16]:
            distance_aggregate[5] += value / 3
        else:
            print("something gone wrong")
    return distance_aggregate
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

def create_feature_counts(n):
    '''
    Utility function to create an empty container in the format required for the feature importance calculations

    Arguments
    ---------
    n: int
        number of classes, generating one unique row for each.

    Returns
    -------
    Dict(Dict)
        An empty container ready for feature importance aggregation.
    '''
    features = ['head', 'left_arm', 'right_arm', 'left_leg', 'right_leg', 'torso']
    feature_counts = {i: {feature: 0 for feature in features} for i in range(n)}
    return feature_counts

def unsupervised_cluster_assessment(input, output, epochs = 15, num_classes = 3, normal_class = 0):
    '''
    Main function for running n k-means models and calculating the mean importances of groups of features.

    Arguments
    ---------
    input: str
        file string for the input data
    output: str
        file string location for the outputted results
    epochs: int (optional, default = 15)
        number of experiments to run to get the mean from
    num_classes: int (optional, default = 3)
        number of classes in the data
    normal_class: int (optional, default = 0)
        Index in the data denoting "normal" gait. Usually but not always the first example

    Returns
    -------
    Pandas.Dataframe, K_means.model, dict(dict), dict(dict)
        returns the results in a dataframe, the latest k-means model for further analysis, the mapping dictionary for the cluster-to-class
        and the feature importances 
    '''
    data, _ = Utilities.process_data_input(input, None)
    data = stitch_data_for_kmeans(data)
    Utilities.save_dataset(data, './code/datasets/joint_data/embed_data/2_people_fixed')
    feature_counts = create_feature_counts(num_classes)
    centroid_distances = []
    for i in range(epochs):
        cluster_percentages, distances, k_model, cluster_map = k_means_experiment(data, num_classes=num_classes, normal_class=normal_class) # used to be clust1,2,3
        print("whats this here: ", cluster_percentages)

        for j in range(len(feature_counts)):
            for k in range(len(cluster_percentages[j])):
                feature_counts[j][cluster_percentages[j][k][0]] += cluster_percentages[j][k][1]/epochs
            
        #print("and the end: ", feature_counts)
        #stop = 5/0
        centroid_distances.append(distances)

    #averages = calculate_column_averages(centroid_distances)
    feature_counts = {
        outer_k: {inner_k: round(inner_v, 2) for inner_k, inner_v in outer_v.items()}
        for outer_k, outer_v in feature_counts.items()
    }
    feature_counts = {
        k: {ik: iv for ik, iv in sorted(v.items(), key=lambda item: item[1], reverse=True)}
        for k, v in feature_counts.items()
    }

    #process data for k-means experiments
    data  = pd.DataFrame(data)
    features = data.iloc[:, 6:].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = apply_grouped_pca(pd.DataFrame(features))
    result_df = predict_and_calculate_proximity(k_model, features, data.iloc[:, :6], cluster_map, feature_counts, normal_class)
    print("cluster map", cluster_map)
    #Add column names
    Utilities.save_dataset(result_df, output)
    return result_df, k_model, cluster_map, feature_counts

def calculate_distances(centroid_A, centroid_B, point_x, weights):
    '''
    Utility function to calculate the distance of an instance from both the normal class and its designated class.

    Arguments
    ---------
    centroid_A: List(float)
        Co-ordinates of the normal centroid
    centroid_B: List(float)
        Co-ordinates of the designated cluster centroid of point_x
    point_x: List(float)
        The instance being examined
    weights: List(float)
        Importance scores for each dimension


    Returns
    -------
    Float, Float
        The distances between A and B and point x in euclidean terms.
    '''
    # Calculate Euclidean distance from point_x to centroid_A
    #the case for comparing the relative distance from normal to abnormal
    d_A = np.linalg.norm(np.array(point_x) - np.array(centroid_A))
    #print("what's this weights values dimwise score: ", weights)
    #converted_weights = convert_region_to_joints(weights)
    #print("now? ", converted_weights)
    #d_A = sum(np.abs(np.array(point_x) - np.array(centroid_A)) * np.array(converted_weights))
    # Calculate Euclidean distance from point_x to centroid_B
    d_B = np.linalg.norm(np.array(point_x) - np.array(centroid_B))
    #d_B = sum(np.abs(np.array(point_x) - np.array(centroid_B)) * np.array(converted_weights))
    print("distances: ", d_A, d_B)
    return d_A, d_B

def convert_region_to_joints(region_values):
    joints = [0 for i in range(18)]
    for ind, val in enumerate(joints):
        if ind in [1,2,3,4]: #1,2,3,4
            joints[ind] = region_values['head']# / 4
        elif ind in [0,17]: #0,17
            joints[ind] = region_values['torso']#  / 2
        elif ind in [5, 7,9]:
            joints[ind] = region_values['left_arm']#  / 3
        elif ind in [6, 8,10]:
            joints[ind] = region_values['right_arm']# / 3
        elif ind in [11,13,15]:
            joints[ind] = region_values['left_leg']#  / 3
        elif ind in [12,14,16]:
            joints[ind] = region_values['right_leg']# / 3
        else:
            print("something gone wrong")
    return joints

def calculate_percentage_closer(d_A, d_B):
    '''
    Utility function to calculate which of the two centroids are closer, and assigning values accordingly.

    Arguments
    ---------
    d_A: float
        distance from the normal centroid
    d_b: float
        distance from the designated centroid

    Returns
    -------
    str, float
        returns which of the two the point is closer to, and by what percentage ratio.
    '''
    # Determine the closer centroid and calculate the percentage closer
    if d_A < d_B:
        d_min = d_A
        d_max = d_B
        closer_to = "A"
    else:
        d_min = d_B
        d_max = d_A
        closer_to = "B"
    percentage_closer = (d_max/ (d_max + d_min)) * 100
    percentage_further = (d_min / (d_max + d_min)) * 100
    return closer_to, percentage_closer

def convert_to_percentage(data_dict):
    '''
    Utility function to relativize feature importance scores so they add up to 1

    Arguments
    ---------
    data_dict: Dict(dict)
        dictionary of feature importances

    Returns
    -------
    Dict(Dict)
        the same dictionary with their values replaced from raw values to percentage values
    '''
    # Calculate the total sum of all values
    percentage_dict = {}
    for i, (key, sub_dict) in enumerate(data_dict.items()):
        total_sum = sum(sub_dict.values())
        # Create a new dictionary with percentage values
        sub_p_dict = {k: (v / (total_sum + 0.0001)) * 100 for k, v in sub_dict.items()}
        percentage_dict[i] = sub_p_dict
    return percentage_dict

def list_immediate_subfolders(folder_path, limit):
    '''
    Utility function to extract n subfolders of images to prevent loading all or any of them depending on use case.

    Arguments
    ---------
    folder_path: str
        Root directory for the images associated to the dataset if they exist.
    limit: int
        The number of subfolders to include

    Returns
    -------
    List(str)
        the list of selected subfolders by their path
    '''
    subfolders = []
    for name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, name)
        if os.path.isdir(full_path):
            subfolders.append(full_path)
        if len(subfolders) >= limit:
            break
    return subfolders

def predict_and_display(data, embed_data, image_data, limit, num_classes = 3, normal_class = 0, dataset_name = 'weightgait'):
    '''
    Function to carry out and (in future) display examples of gait sequences with their predicted cluster and said gaits level of 
    severity.

    Arguments
    ---------
    data: List(List)
        un-changed input data prior to embedding or other transformation
    embed_data: List(List)
        corresponding data after being passed into embedding space by ST-TAGCN
    image_data: List(List)
        Image data to display along with examples if needed
    num_classes: int (optional, default = 3)
        Number of classes exhibited in the data
    normal_class: int (optional, default = 0)
        Index in the data denoting "normal" gait. Usually but not always the first example
    dataset_name: str (optional, default = 'weightgait')
        folder to save the results to

    Returns
    -------
    None
    '''
    #First load in the raw data of 5 people
    sub_folders = list_immediate_subfolders(image_data, limit)
    #load in corresponding embedding data 
    embed_joints, _ = Utilities.process_data_input(embed_data, None)

    #split the raw data and images into blocks of 7 the same way the embedding data is set up
    predictions, model, cluster_map, feature_counts = unsupervised_cluster_assessment(embed_joints, './code/datasets/joint_data/embed_data/proximities',
                                                                                       epochs= 20, num_classes=num_classes, normal_class=normal_class)
    segmented_joints = stitch_data_for_kmeans(embed_joints)

    segmented_joints  = pd.DataFrame(segmented_joints)
    features = segmented_joints.iloc[:, 6:].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    segmented_joints = apply_grouped_pca(pd.DataFrame(features))
    centroids = model.cluster_centers_
    #calculate their relative distance to both the normal centroid and the prediction centroid 
    #predict confidence by inverse closeness (if it's 75% closer to 1 than 0, its 75% confidence), raw distance from normal centroid is severity, 
    #compare with outermost range of normal and innermost
    #print the importance vectors for this cluster and the individual example
    
    #draw the person with their skeleton
    #black_frame = [[0 for _ in inner_list] for inner_list in raw_images]
    dimwise_scores = convert_to_percentage(feature_counts)
    closeness_preds = []
    for i in range(len(segmented_joints)):
        #calculate severity here:
        #percentage score of importance * 
        print("\nPrediction for this clip is class: ", predictions[i][-2])
        print("Predicted severity ", predictions[i][-1])
        #importance vector for this cluster here printed
        #calculate distance from both centroids 
        if cluster_map[normal_class] != cluster_map[predictions[i][-2]]:
            d_a, d_b = calculate_distances(centroids[cluster_map[normal_class]], centroids[cluster_map[predictions[i][-2]]], segmented_joints.iloc[i], dimwise_scores[predictions[i][-2]])
        else:
            d_a, d_b = calculate_distances(centroids[cluster_map[normal_class]], centroids[cluster_map[normal_class + 1]], segmented_joints.iloc[i], dimwise_scores[predictions[i][-2]])
        closeness = calculate_percentage_closer(d_a, d_b)
        closeness_preds.append(closeness)
        print("Confidence as a percentage: ", closeness if closeness != 0 else 1)
        print("what's this here: ", predictions[i][-2])
        print("dimwise scores: ", dimwise_scores, len(dimwise_scores))
        print("\nThe DIMWISE scores for this cluster are: ", dimwise_scores[predictions[i][-2]])


    print("all dimwise scores for clusters: ", dimwise_scores)
    print("cluster map: ", cluster_map)
    for row, new_value in zip(predictions, closeness_preds):
        row.append(new_value[1])
    #column names are: [instance, no_in_instance, class, freeze, obstacle, person, cluster_prediction, severity_prediction, confidence_prediction ] 9 items
    df = pd.DataFrame(predictions, columns=['instance', 'no_in_instance', 'class', 'freeze', 'obstacle', 'person',
                                             'cluster_prediction', 'severity_prediction', 'confidence_prediction'])
    print(df.head(3))
    grouped = df.groupby('class')
    # Dictionary to store the mean DataFrame for each group
    mean_dfs = {}
    # Iterate over each group and calculate the mean
    for name, group in grouped:
        mean_dfs[name] = group.mean()
    # Display the result for each group
    for key, value in mean_dfs.items():
        print(f"Group {key}:\n{value}\n")

    Utilities.save_dataset(predictions, f'./Code/Datasets/Joint_Data/Results/{dataset_name}')
    print("this: ", {key: df['confidence_prediction'].tolist() for key, df in mean_dfs.items()})
    return {key: df['confidence_prediction'].tolist() for key, df in mean_dfs.items()}