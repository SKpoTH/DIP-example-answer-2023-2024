import numpy as np
import cv2 as cv

from MyDIP.filters.smoothing import gaussianFilter

def flatfieldCorrection(input_img, gauss_size, dark_img=None):
    '''
        Flat Field Correction, Illumination Normalization
    '''
    # - Convert to float, operate with division/multiplication
    input_img = input_img.astype(float)
    # - Dark Image is not defined
    if dark_img == None:
        dark_img = np.zeros_like(input_img)

    ### -> Flat-field image by Gaussian Filter
    gauss_filter = gaussianFilter(gauss_size)
    ff_img = cv.filter2D(input_img, -1, gauss_filter)

    ### -> Mean of Flat-field image
    ff_mean = np.mean(ff_img)
    ### -> Correction
    output_img = (ff_mean / (ff_img - dark_img)) * (input_img - dark_img)

    # -> Clip range into [0, 255]
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)

    return output_img