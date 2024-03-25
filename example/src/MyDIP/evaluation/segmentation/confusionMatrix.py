import numpy as np
from sklearn.metrics import confusion_matrix

def confusionMatrix(output_img, gt_img):
    '''
        Confusion Matrix of Segmentation
    '''
    ### -> Create Main Area of Confusion Matrix
    matrix = confusion_matrix(gt_img.flatten(), output_img.flatten()).T
    
    # -> Expand Matrix for Precision & Recall
    con_matrix = np.zeros((matrix.shape[0]+1, matrix.shape[1]+1))
    con_matrix[:2, :2] = matrix
    # -> Precision
    prec_bg = matrix[0, 0] / np.sum(matrix[0, :])
    prec_fg = matrix[1, 1] / np.sum(matrix[1, :])
    con_matrix[:2,-1] = [prec_bg, prec_fg]
    # -> Recall
    rec_bg = matrix[0, 0] / np.sum(matrix[:, 0])
    rec_fg = matrix[1, 1] / np.sum(matrix[:, 1])
    con_matrix[-1,:2] = [rec_bg, rec_fg]
    # -> Accuracy
    accuracy = (matrix[0, 0] + matrix[1, 1]) / np.sum(matrix)
    con_matrix[-1, -1] = accuracy

    return con_matrix