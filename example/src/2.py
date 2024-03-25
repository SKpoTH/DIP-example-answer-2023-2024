import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.morphology as skmorph
from skimage.exposure import equalize_hist, equalize_adapthist
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks

from glob import glob

# <-------------------------------> My DIP <--------------------------------->
from MyDIP.intensityTransform import *
from MyDIP.fourier import Fourier2D
from MyDIP.filters import *
from MyDIP.general import adjustRange
# <----------------------------> Sub Program <------------------------------->
def houghCircles(edge_img, vote_thresh, radius_list, input_img):
    '''
        Circle Hough Transform
    '''

    # -> Hough Circle Transform
    hough_res = hough_circle(edge_img, radius_list)

    # -> Hough Circle Detection Cutoff
    votes, center_x, center_y, radius = hough_circle_peaks(hough_res, 
                                                           radius_list,  
                                                           min_xdistance=10,  
                                                           min_ydistance=10)

    output_circle = []
    for v, cx, cy, r in zip(votes, center_x, center_y, radius):
        if v >= vote_thresh:
            output_circle.append((v, cx, cy, r))

    # ~> Hough Visualization
    input_img = input_img.astype(np.uint8)
    output_img = cv.cvtColor(input_img, cv.COLOR_GRAY2RGB)

    # ~> Draw Ciccle
    for i in range(len(output_circle)):
        v, cx, cy, r = output_circle[i]
        cv.circle(output_img, (cx, cy), r, (0, 0, 255), 1)

    return output_circle, output_img

# <-------------------------------> PATH <----------------------------------->
DATASET_PATH = "dataset/2/"
INPUT_PATH = DATASET_PATH

# <---------------------------> Main Program <------------------------------->
if __name__ == "__main__":
    # -> Get file paths
    input_files = glob(INPUT_PATH + "*")

    # -> Create Output destination if not created yet
    output_path = "output/2/"
    if not os.path.exists(output_path): 
        os.makedirs(output_path)

    for i in range(0, len(input_files)):
        ### -> Read image
        input_filename = os.path.basename(input_files[i])
        # - Input
        input_img = cv.imread(input_files[i])

        ### -> Convert to Grayscale
        gray_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)

        ### -> Canny Edge Detection
        # - Gaussian Blur Parameter
        blur_size = 35
        blur_sigma = 0.3*((blur_size-1)*0.5-1)+0.8
        # - Appy Canny
        edge_img = canny(gray_img, blur_sigma,
                         low_threshold=0.75, high_threshold=0.9,
                         use_quantiles=True, mode="mirror")

        ### -> Circle Hough Transform
        radius_list = np.arange(7, 30)
        output_circle, output_img = houghCircles(edge_img, 0.35, radius_list, gray_img)

        # ~> Display
        plt.subplot(1, 2, 1)
        plt.imshow(gray_img, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(output_img)
        plt.show()