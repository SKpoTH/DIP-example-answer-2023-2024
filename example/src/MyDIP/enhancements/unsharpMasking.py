import numpy as np
import cv2 as cv

from MyDIP.filters.smoothing import gaussianFilter

def unsharpMasking(input_img, blurfilter_size, k):
    '''
        Sharpening by Unsharp Masking, High Boosting Technique
    '''
    # - Convert to float, operate multiplying
    input_img = input_img.astype(float)
    ### -> Gaussian Filtering, Low components
    gauss_filter = gaussianFilter(blurfilter_size)
    low_img = cv.filter2D(input_img, -1, gauss_filter)

    ### -> Unsharp Masking, High components
    high_img = input_img - low_img
    ### -> High Boosting
    output_img = input_img + (k * high_img)
    ### -> Clip range into [0, 255]
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)

    return output_img