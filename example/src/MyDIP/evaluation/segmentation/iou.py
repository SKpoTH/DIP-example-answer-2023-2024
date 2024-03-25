import numpy as np

def iou(output_img, gt_img):
    '''
        Intersection over Union
    '''
    # -> Intersection (AND operation)
    intersect = np.logical_and(output_img, gt_img)
    # -> Union (OR operation)
    union = np.logical_or(output_img, gt_img)
    ### -> Intersection over Union (IoU)
    IoU = np.sum(intersect) / np.sum(union)

    return IoU

