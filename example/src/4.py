import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.morphology as skmorph
from skimage.exposure import equalize_hist
from skimage.feature import canny
from skimage.measure import regionprops
from glob import glob

# <-------------------------------> My DIP <--------------------------------->
from MyDIP.fourier import Fourier2D
from MyDIP.filters import *
from MyDIP.general import adjustRange
from MyDIP.segmentations import kmeans
from MyDIP.morphology import *
from MyDIP.evaluation.segmentation import *
# <----------------------------> Sub Program <------------------------------->

# <-------------------------------> PATH <----------------------------------->
DATASET_PATH = "dataset/4/"
INPUT_PATH = DATASET_PATH + "fish/"
SEGMENT_PATH = DATASET_PATH + "fish_gt/"

# <---------------------------> Main Program <------------------------------->
if __name__ == "__main__":
    # -> Get file paths
    input_files = glob(INPUT_PATH + "*")
    gt_files = glob(SEGMENT_PATH + "*")

    # -> Create Output destination if not created yet
    output_path = "output/4/"
    if not os.path.exists(output_path): 
        os.makedirs(output_path)

    for i in range(0, len(input_files)):
        ### -> Read image
        input_filename = os.path.basename(input_files[i])
        # - Input
        input_img = cv.imread(input_files[i])
        red_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)[:,:,0]
        # - Ground Truth
        gt_img = cv.cvtColor(cv.imread(gt_files[i]), cv.COLOR_BGR2GRAY)
        gt_img = np.where(gt_img > 130, 1, 0)

        ### -> Feature Extraction
        obj_list = regionprops(gt_img, intensity_image=red_img)

        x_feature = []
        for obj in obj_list:
            # -> Red Color
            intensity_mean = obj.intensity_mean
            # print(obj.intensity_mean)

            # -> Minor Axis Length
            minor_axis = obj.axis_minor_length
            # print(obj.axis_minor_length)

            # -> Circularity
            circular = (4*np.pi*obj.num_pixels) / (obj.perimeter ** 2)
            # print(circular)

            # -> Append to vector
            x_feature.extend([intensity_mean, circular, minor_axis])
        
        ### -> Classifier
        fish_class = "?"

        if x_feature[0] > 210:
            fish_class = "fish_A"
        elif x_feature[1] > 0.4:
            fish_class = "fish_C" 
        else:
            if x_feature[2] < 100:
                fish_class = "fish_D"
            else:
                fish_class = "fish_B"

        ### -> Class Answer
        print(fish_class)
