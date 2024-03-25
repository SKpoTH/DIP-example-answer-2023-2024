import cv2 as cv
import skimage.morphology as skmorph

def boundaryExtract(input_img):
    '''
        Find Boundary of a Binary Image
    '''
    # - Erosion
    stre = skmorph.disk(1)
    erode_img = cv.erode(input_img, stre)
    # - Subtraction
    output_img = input_img - erode_img

    return output_img


