import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.morphology as skmorph
from skimage.exposure import equalize_hist
from skimage.feature import canny
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
DATASET_PATH = "dataset/3/"
INPUT_PATH = DATASET_PATH + "fish/"
SEGMENT_PATH = DATASET_PATH + "fish_gt/"

# <---------------------------> Main Program <------------------------------->
if __name__ == "__main__":
    # -> Get file paths
    input_files = glob(INPUT_PATH + "*")
    gt_files = glob(SEGMENT_PATH + "*")

    # -> Create Output destination if not created yet
    output_path = "output/3/"
    if not os.path.exists(output_path): 
        os.makedirs(output_path)

    for i in range(0, len(input_files)):
        ### -> Read image
        input_filename = os.path.basename(input_files[i])
        # - Input
        input_img = cv.imread(input_files[i])
        # - Ground Truth
        gt_img = cv.cvtColor(cv.imread(gt_files[i]), cv.COLOR_BGR2GRAY)
        gt_img = np.where(gt_img > 130, 1, 0)

        # ### -> Easy Segmentation
        # rgb_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)
        # _, thresh_img = cv.threshold(rgb_img[:,:,0], 170, 255, cv.THRESH_BINARY) 

        ### -> Segmentation
        # - Convert to HSV
        hsv_img = cv.cvtColor(input_img, cv.COLOR_BGR2HSV)
        # - K-means Clustering
        kmean_img, _ = kmeans(hsv_img, k=4)
        # - Cluster's red color thresholding
        rgb_img = cv.cvtColor(kmean_img, cv.COLOR_HSV2RGB)
        _, thresh_img = cv.threshold(rgb_img[:,:,0], 165, 255, cv.THRESH_BINARY)
        # - Morphology
        morph_img = fragmentsRemove(thresh_img, 0.02)
        output_img = fillHoles(morph_img)

        ### -> IoU
        print(iou(output_img, gt_img))

        # # ~> Display
        # plt.subplot(1, 2, 1)
        # plt.imshow(rgb_img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(output_img, cmap="gray")
        # plt.show()

        # -> Save image
        output_file = output_path + input_filename
        cv.imwrite(output_file, output_img*255)