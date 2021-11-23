import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict

dataset = pd.read_pickle('sample_dataset_kmeans.pickle')
centroids = pd.read_pickle('sample_centroids_kmeans.pickle')

# Problem 1
def k_means_clustering(centroids, dataset):
    #   Description: Perform k means clustering for 2 iterations given as input the dataset and centroids.
    #   Input:
    #       1. centroids - A list of lists containing the initial centroids for each cluster.
    #       2. dataset - A list of lists denoting points in the space.
    #   Output:
    #       1. results - A dictionary where the key is iteration number and store the cluster assignments in the
    #           appropriate clusters. Also, update the centroids list after each iteration.

    result = {
        '1': {'cluster1': [], 'cluster2': [], 'cluster3': [], 'centroids': []},
        '2': {'cluster1': [], 'cluster2': [], 'cluster3': [], 'centroids': []}
    }

    centroid1, centroid2, centroid3 = centroids[0], centroids[1], centroids[2]

    for iteration in range(2):
        # your code here
        for point in dataset:
            euc_dist = []
            for centroid in centroids:
                euc_dist.append(((point[0]-centroid[0])**2 + (point[1]-centroid[1])**2)**(0.5))
            cluster = euc_dist.index(min(euc_dist))
            if cluster == 0:
                result[f'{iteration+1}']['cluster1'].append(point)
            elif cluster == 1:
                result[f'{iteration+1}']['cluster2'].append(point)
            else:
                result[f'{iteration+1}']['cluster3'].append(point)
        x = []
        y = []
        for point in result[f'{iteration+1}']['cluster1']:
            x.append(point[0])
            y.append(point[1])
        centroid1 = [sum(x)/len(x), sum(y)/len(y)]
        x = []
        y = []
        for point in result[f'{iteration + 1}']['cluster2']:
            x.append(point[0])
            y.append(point[1])
        centroid2 = [sum(x) / len(x), sum(y) / len(y)]
        x = []
        y = []
        for point in result[f'{iteration + 1}']['cluster3']:
            x.append(point[0])
            y.append(point[1])
        centroid3 = [sum(x) / len(x), sum(y) / len(y)]
        centroids = [centroid1, centroid2, centroid3]
        result[f'{iteration + 1}']['centroids'] = [centroid1, centroid2, centroid3]
    return result

dataset = pd.read_pickle('sample_dataset_em.pickle')
centroids = pd.read_pickle('sample_centroids_em.pickle')
print(dataset)
print(centroids)

def em_clustering(centroids, dataset):
    #   Input:
    #       1. centroids - A list of lists with each value representing the mean and standard deviation values picked from a gausian distribution.
    #       2. dataset - A list of points randomly picked.
    #   Output:
    #       1. results - Return the updated centroids(updated mean and std values after the EM step) after the first iteration.

    new_centroids = list()

    # your code here
    mu1 = centroids[0][0]
    mu2 = centroids[1][0]
    sig1 = centroids[0][1]
    sig2 = centroids[1][1]

    # E Step
    p1 = []
    p2 = []
    for x in dataset:
        p1_non = (1 / (sig1 * np.sqrt(2 * np.pi))) * np.exp(-(1 / 2) * (((x - mu1) / sig1) ** 2))
        p2_non = (1 / (sig2 * np.sqrt(2 * np.pi))) * np.exp(-(1 / 2) * (((x - mu2) / sig2) ** 2))
        p1.append(p1_non/(p1_non+p2_non))
        p2.append(p2_non/(p1_non+p2_non))
    sum_p1 = sum(p1)
    sum_p2 = sum(p2)

    # M step
    mu_new1 = 0
    mu_new2 = 0
    for i in range(len(dataset)):
        mu_new1 = mu_new1 + ((p1[i] * dataset[i])/sum_p1)
        mu_new2 = mu_new2 + ((p2[i] * dataset[i])/sum_p2)
    sig_new1 = 0
    sig_new2 = 0
    for i in range(len(dataset)):
        sig_new1 = sig_new1 + (p1[i] * ((dataset[i] - mu_new1) ** 2))
        sig_new2 = sig_new2 + (p2[i] * ((dataset[i] - mu_new2) ** 2))
    sig_new1 = np.sqrt(sig_new1)/np.sqrt(sum_p1)
    sig_new2 = np.sqrt(sig_new2)/np.sqrt(sum_p2)

    new_centroids.append([mu_new1, sig_new1])
    new_centroids.append([mu_new2, sig_new2])
    return new_centroids

result = pd.read_pickle('sample_result_em.pickle')
print(result)
print(em_clustering(centroids, dataset))
