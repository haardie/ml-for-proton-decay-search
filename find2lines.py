#!~/venv/bin/python

import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import scipy.sparse as sp
from skimage.measure import label, regionprops
from skimage.segmentation import flood_fill

"""
This script finds track continuation cases ("two lines") in the images (sparse matrices)
"""


def split_path(path):
    """
    Split the path of file into type, current_plane, and file number
    :param path:  path of the file
    :return:    type (signal or background), current_plane, file number
    """
    filename = os.path.basename(path)
    if 'larcv' in filename:
        type = filename.split('_')[1]
        batch = filename.split('_')[2]
        plane = int(filename.split('_')[4][5:])
        file_num = int(filename.split('_')[5].split('.')[0])

        return type, batch, plane, file_num

    else:
        type = filename.split('_')[0]
        plane = int(filename.split('_')[1][5:])
        file_num = int(filename.split('_')[2].split('.')[0])

        return type, 0, plane, file_num


def get_image_dict(path_list):
    """
    Get a dictionary of images from a directory of sparse matrices
    :param path_list:   list of paths of sparse matrices
    :return:
    """
    image_dict = {}

    for path in path_list:
        if path.startswith('.DS'):
            continue
        if path.endswith('.npz'):
            # Extract key from filename
            key = path.split('.')[0]
            image = sp.load_npz(path)
            image = image.toarray()
            image = np.uint8(image)
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            image_dict[key] = image
    return image_dict


def constrained_flood(image_dict, min_reg_area, max_vert_dist, tolerance=10):
    """
    Find two lines in the images using constrained flood fill

    :param image_dict:  dictionary of images
    :param min_reg_area:    minimum area of a region to be considered
    :param max_vert_dist:   maximum vertical distance between two regions to be considered "one line"
    :param tolerance:   tolerance for flood fill (how different the pixel intensity can be from the seed pixel)
    :return:    list of paths of images with "two lines"
    """
    two_line_paths = []

    for key, img in image_dict.items():

        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        _, binary_mask = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY)
        nonzero_pixels = np.transpose(np.nonzero(binary_mask))

        if len(nonzero_pixels) == 0:
            continue

        # Choose a random seed pixel
        rand_pix_idx = np.random.choice(len(nonzero_pixels))

        x_seed, y_seed = nonzero_pixels[rand_pix_idx]

        labeled_mask = flood_fill(binary_mask, (x_seed, y_seed), 255, tolerance=tolerance)
        labeled_img = label(labeled_mask)

        regions = regionprops(labeled_img)
        region_centroids = {}
        region_pix_coords = {}

        # Create a mask for large regions
        large_region_mask = np.zeros_like(labeled_img, dtype=bool)

        found_two_lines = False

        # Find large regions and their centroids
        for region in regions:
            if region.area >= min_reg_area:
                large_region_mask[labeled_img == region.label] = True
                region_centroids[region.label] = region.centroid
                region_pixels = np.transpose(np.nonzero(labeled_img == region.label))
                region_pix_coords[region.label] = region_pixels

        for label1, centroid1 in region_centroids.items():
            for label2, centroid2 in region_centroids.items():
                if label1 != label2:
                    regA = np.transpose(np.nonzero(labeled_img == label1))
                    regB = np.transpose(np.nonzero(labeled_img == label2))

                    # Calculate the vertical distances between all pairs of points in the regions
                    vert_differences = np.abs(regA[:, 0][:, np.newaxis] - regB[:, 0])
                    vert_dist = np.min(vert_differences)

                    if vert_dist >= max_vert_dist:
                        found_two_lines = True
                        break

            if found_two_lines:
                break
            """
            fig = plt.figure(figsize=(10, 10), dpi=200)
            plt.imshow(img, interpolation='nearest', aspect='auto', cmap='gray')
            for region in regionprops(labeled_img):
                if region.area >= min_reg_area:
                    minr, minc, maxr, maxc = region.bbox
                    rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                         fill=False, edgecolor='red', linewidth=1)
                    ax = plt.gca()
                    ax.add_patch(rect)
            plt.tight_layout()
            plt.show()
            """

    return two_line_paths


data_dir = '/mnt/lustre/helios-home/gartmann/pdecay_sorted/K_pi0pi0pi/'
plane0 = os.path.join(data_dir, f'plane0')
plane1 = os.path.join(data_dir, f'plane1')

evt_files0 = [os.path.join(plane0, file) for file in os.listdir(plane0) if '.npz' in file]
evt_files1 = [os.path.join(plane1, file) for file in os.listdir(plane1) if '.npz' in file]

print(evt_files0[:10])
print(evt_files1[:10])

two_lines0 = constrained_flood(image_dict=get_image_dict(path_list=evt_files0), min_reg_area=400, max_vert_dist=400,
                               tolerance=10)
two_lines1 = constrained_flood(image_dict=get_image_dict(path_list=evt_files1), min_reg_area=400, max_vert_dist=400,
                               tolerance=10)

print(
    f'Found {len(two_lines0)}/{len(evt_files0)} or {len(two_lines0) / len(evt_files0) * 100:.2f}% double tracks in plane0')
print(
    f'Found {len(two_lines1)}/{len(evt_files1)} or {len(two_lines1) / len(evt_files1) * 100:.2f}% double tracks in plane1')
print(
    f'Totally found {len(two_lines0) + len(two_lines1)}/{len(evt_files0) + len(evt_files1)} or {(len(two_lines0) + len(two_lines1)) / (len(evt_files0) + len(evt_files1)) * 100:.2f}% double tracks')

# Remove the files with the double tracks
for path in two_lines0:
    type, batch, plane, file_number = split_path(path)
    path1 = os.path.join(data_dir, f'plane1', f'files_{type}_{batch}_larcv_plane1_{file_number}.npz')
    path2 = os.path.join(data_dir, f'plane2', f'files_{type}_{batch}_larcv_plane2_{file_number}.npz')

    if os.path.exists(path1) and os.path.exists(path2):
        os.remove(path)
        os.remove(path1)
        os.remove(path2)

    else:
        os.remove(path)

for path in two_lines1:
    type, batch, plane, file_number = split_path(path)

    path0 = os.path.join(data_dir, f'plane0', f'files_{type}_{batch}_larcv_plane0_{file_number}.npz')
    path2 = os.path.join(data_dir, f'plane2', f'files_{type}_{batch}_larcv_plane2_{file_number}.npz')

    if os.path.exists(path0) and os.path.exists(path2):
        os.remove(path)
        os.remove(path0)
        os.remove(path2)

    else:
        os.remove(path)
