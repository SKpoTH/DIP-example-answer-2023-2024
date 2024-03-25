import cv2 as cv
import numpy as np
from MyDIP.general import adjustRange

def detector(input_img, filter, thresh_val, abs_res=False):
    '''
        Shape Detector by given filter and cutoffs
    ''' 
    # - Convert to float, operate with a negative number
    input_img = input_img.astype(float)
    # -> Filtering
    res_img = cv.filter2D(input_img, -1, filter)
    if abs_res:
        res_img = abs(res_img)
    # -> Convert range to [0, 1]
    res_img = adjustRange(res_img, (res_img.min(), res_img.max()), (0, 1))
    # -> Cutoff Detected response
    output_img = np.where(res_img > thresh_val, 255, 0)
    output_img = output_img.astype(np.uint8)

    return output_img