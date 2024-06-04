import pickle
import numpy as np

def load_data_from_pickle(filepath):
    # Open the pickle file in binary read mode
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def scale_and_offset(data, errors):
    mean_data = np.mean(data)
    std_data = np.std(data)
    std_errors = np.std(errors)

    data_scale = 1 / std_data
    error_scale = 1 / std_errors

    data_offset = -mean_data * data_scale

    return data_scale, error_scale, data_offset
