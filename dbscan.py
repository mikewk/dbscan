import math
import re

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


# Function to quickly turn debug print statements off and on
def print_debug(string):
    # print(string)
    return


def db_scan(filename, eps, min_samples, min_unique, min_gap):
    print(filename)
    data = pd.read_csv(filename, header=None, parse_dates=[2])  # Read file in with pandas to get proper date times
    data[3] = data[2].dt.normalize()  # Create date column to collate by day
    dates = data[3].unique()
    output = dict()
    for date in dates:
        X = data[data[3] == date]  # Get all data for this date
        X = X[[0, 1]]  # Only send the Lat/Long to DBScan
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_  # These are the cluster labels for each data point
        print_debug(labels)
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        n_points_ = len(labels)

        # print("Estimated number of clusters: %d" % n_clusters_)
        # print("Estimated number of noise points: %d" % n_noise_)
        # print("Total number of points: %d" % n_points_)

        # Setup variables to calculate entropy over labels AND time
        previous_label = -1
        count = 0
        noise_count = min_gap + 1
        last_good_label = -1
        current_unique_num = 0

        unique_labels = dict()
        for label in labels:
            # If we have a change
            if label != previous_label:
                print_debug("found change")
                # If the previous label was a noise block
                if previous_label == -1:
                    print_debug("last label noise")
                    # if the previous label was noise and big enough, then this is just simply a new section
                    if noise_count >= min_gap:
                        print_debug("new section")
                        # if the last good block was big enough, save it
                        if count >= min_unique:
                            print_debug("save old section big enough")
                            unique_labels[current_unique_num] = count
                            current_unique_num += 1
                        previous_label = label
                        noise_count = 0
                        count = 1
                        last_good_label = label
                    # if the previous label was noise but less than the min gap, check to see if this is a continuation
                    elif last_good_label == label:
                        print_debug("continuation")
                        count += 1
                        previous_label = label
                        noise_count = 0
                    # If the previous label was noise, less than min gap, and the last block was good, save it
                    elif last_good_label != -1:
                        print_debug("check big enough")
                        if count >= min_unique:
                            print_debug("save section big enough")
                            unique_labels[current_unique_num] = count
                            current_unique_num += 1
                        last_good_label = label
                        previous_label = label
                        count = 0
                        noise_count = 0
                else:
                    # If the current label is a noise block
                    if label == -1:
                        print_debug("new noise block")
                        # then we need to wait to see the current block will continue
                        last_good_label = previous_label
                        noise_count = 1
                        previous_label = label
                    else:
                        # If the new block is not noise, then save the old block if it's big enough
                        print_debug("new location, check if old wsa big enough")
                        if count > min_unique:
                            print_debug("old big enough")
                            unique_labels[current_unique_num] = count
                            current_unique_num += 1
                        count = 1
                        previous_label = label
                        last_good_label = label
            else:
                # If we have the same label
                if label == -1:
                    noise_count += 1
                else:
                    count += 1
        # then, finally, clean up the last block
        if last_good_label != -1 and count > min_unique:
            unique_labels[current_unique_num] = count

        print_debug(unique_labels)

        # calculate entropy
        locations = np.fromiter(unique_labels.values(), dtype=float)
        print(locations)
        denom = sum(locations)
        prob = locations / denom
        inverse_prob = 1 / prob
        logp = np.log2(inverse_prob)
        plogp = prob * logp
        sum_plogp = np.sum(plogp)
        entropy = sum_plogp / np.log2(denom)
        print_debug(entropy)
        output[date.date()] = entropy
    return output
