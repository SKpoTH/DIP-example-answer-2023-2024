import cv2 as cv
import numpy as np
import skimage.morphology as skmorph

def reconstruction(seed_img, mask_img, iteration, method, stre_size=3):
    '''
        Morphological Reconstruction
    '''
    # -> Set initial image
    init_img = seed_img.copy()

    for i in range(iteration):
        # -> Dilation Case
        if method == "dilation":
            # -> Dilation
            stre = skmorph.disk(stre_size)
            dilate_img = cv.dilate(init_img, stre)
            # -> Compare Mask, Intersection, compare min
            init_img = np.minimum(dilate_img, mask_img)
        # -> Erosion Case
        elif method == "erosion":
            # -> Erosion
            stre = skmorph.disk(stre_size)
            erode_img = cv.erode(init_img, stre)
            # -> Compare Mask, Union, compare max
            init_img = np.maximum(erode_img, mask_img)
    
    # -> Current state as output
    output_img = init_img

    return output_img

