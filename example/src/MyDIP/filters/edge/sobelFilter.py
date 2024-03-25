import numpy as np

def sobelFilter(sobel_type):
    '''
        Sobel Filter 3x3
    '''
    if sobel_type == "horizontal":
        sobel_filter = np.array([[ 1, 2, 1],
                                 [ 0, 0, 0],
                                 [-1,-2,-1]])
    elif sobel_type == "vertical":
        sobel_filter = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1,-0, 1]])
    elif sobel_type == "diagonal1":
        sobel_filter = np.array([[ 0, 1, 2],
                                 [-1, 0, 1],
                                 [-2,-1, 0]]) 
    elif sobel_type == "diagonal2":
        sobel_filter = np.array([[ 2, 1, 0],
                                 [ 1, 0,-1],
                                 [ 0,-1,-2]]) 

    return sobel_filter