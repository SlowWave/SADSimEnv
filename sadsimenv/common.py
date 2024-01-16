import numpy as np


def standardization(arr):
    
    # get the mean
    mean_val = np.mean(arr)
    
    # get the standard deviation
    std_val = np.std(arr)
    
    # center data around 0 mean and scale to have 1 std 
    return (arr - mean_val) / std_val