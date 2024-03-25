import numpy as np

def accuracy(output_img, gt_img):
    '''
        Accuracy of Segmentation
    '''
    # -> Compare pixel (XOR and then Invert) 
    comp_pixel = ~np.logical_xor(output_img, gt_img)
    match_pixel = np.sum(comp_pixel)
    # -> Total number of pixels
    img_height, img_width = output_img.shape
    total_pixel = img_height * img_width
    ### -> Accuracy
    acc = match_pixel / total_pixel

    return acc