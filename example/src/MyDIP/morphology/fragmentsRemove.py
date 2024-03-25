import cv2 as cv
import numpy as np

def fragmentsRemove(input_img, thresh_ratio=0.1):
    '''
        Remove Fragments in the Binary Image
    '''
    # -> Connected Components
    _, label_img = cv.connectedComponents(input_img)
    # -> Count Number of pixel of each label
    labels, counts = np.unique(label_img, return_counts=True)

    ### -> Fragments Searching
    # - Cut-off Value
    count_pixels = input_img.shape[0] * input_img.shape[1]
    count_thresh = int(count_pixels * thresh_ratio)
    # - Thresholding
    pass_index = np.argwhere(counts > count_thresh).flatten()
    assert len(pass_index) > 1, "All Objects are Fragments, \
                                 Try reducing 'thresh_ratio' value"
    # - Pass Label/Group
    output_img = np.isin(label_img, pass_index[1:])
    output_img = output_img.astype(np.uint8)

    return output_img


