#!~/my_venv/bin/python

import os
import numpy as np

def get_ROI(dir_path, filename):
    """
    The function extracts the ROI coordinates from the .csv file containing the event data.
    :param dir_path: path to the directory containing the .csv file
    :param filename: name of the .csv file
    :return: ROI matrix, x_min (leftmost x coordinate), x_max (rightmost x coordinate),
    y_min (topmost y coordinate), y_max (bottommost y coordinate)
    """
    try:
        with open(os.path.join(dir_path, filename)) as file:
            rows, cols = map(int, file.readline().split(" x "))
            
            # Reorganize the linear data into a matrix
            matrix = np.fromfile(file, dtype=int, count=cols * rows, sep="\n").reshape((rows, cols))
            
            # Extract the ROI coordinates ans slice the ROI from the matrix
            line = file.readline().rstrip().split()
            x_min = int(line[3])
            x_max = int(line[4])
            
            line = file.readline().rstrip().split()
            y_min = int(line[2])
            y_max = int(line[3])
            roi = matrix[y_min:y_max, x_min:x_max]
            
            # The data is not normalized, and negative values may occur. Set them to 0.
            roi[roi < 0] = 0
    except (ValueError, FileNotFoundError):
        # If the ROI is not found, return None (and None for the coordinates
        return None, None, None, None, None
    return roi, x_min, x_max, y_min, y_max


def center_ROI(roi, dim):
    """
    The function centers the ROI in a square matrix of size dim x dim. Padding and cropping is used.
    :param roi: ROI matrix
    :param dim: size of the square matrix
    :return: centered ROI matrix
    """
    # Case 1: ROI width and height are smaller than dim
    if roi.shape[0] < dim and roi.shape[1] < dim:
            pad_y = (dim - roi.shape[0]) // 2
            pad_x = (dim - roi.shape[1]) // 2
            roi = np.pad(roi, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
    
    # Case 2: ROI width is smaller than dim, but height is larger than dim
    elif roi.shape[0] < dim < roi.shape[1]:
        pad_y = (dim - roi.shape[0]) // 2
        roi = np.pad(roi, ((pad_y, pad_y), (0, 0)), mode='constant')
    
    # Case 3: ROI height is smaller than dim, but width is larger than dim
    elif roi.shape[0] > dim > roi.shape[1]:
        pad_x = (dim - roi.shape[1]) // 2
        roi = np.pad(roi, ((0, 0), (pad_x, pad_x)), mode='constant')
    
    # Case 4: ROI width and height are larger than dim
    elif roi.shape[0] > dim and roi.shape[1] > dim:
        nonzero = np.nonzero(roi)
        y_min = np.min(nonzero[0])
        y_max = np.max(nonzero[0])
        x_min = np.min(nonzero[1])
        x_max = np.max(nonzero[1])
        roi = roi[y_min:y_max, x_min:x_max]

        pad_y = (roi.shape[0] - dim) // 2
        pad_x = (roi.shape[1] - dim) // 2
        roi = np.pad(roi, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')

    return roi

