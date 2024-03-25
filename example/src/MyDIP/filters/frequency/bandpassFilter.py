import numpy as np

def distanceMap(size, pos):
    '''
        Distance Map 2D for Function index
    '''
    map_height = size[0]
    map_width = size[1]
    
    # -> Preset Position Index
    v = np.arange(0.5*((map_height+1)%2), 0.5*((map_height+1)%2)+map_height)
    u = np.arange(0.5*((map_width+1)%2), 0.5*((map_width+1)%2)+map_width)
    uv, vv = np.meshgrid(u, v)

    # -> Distance Map 
    distance_map = ((uv-pos[1])**2 + (vv-pos[0])**2)**0.5

    return distance_map

def idealFunction(distance_map, band_center, band_width):
    '''
        Ideal Function
    '''
    ideal_func = distance_map.copy()
    ideal_func = np.where((distance_map >= (band_center-band_width/2)) &
                          (distance_map <= (band_center+band_width/2)), 
                           1, 0)

    return ideal_func

def gaussianFunction(distance_map, band_center, band_width):
    '''
        Gaussian Function
    '''
    gauss_func = np.exp(-((distance_map**2-band_center**2) / (distance_map*band_width))**2)

    return gauss_func

def butterworthFunction(distance_map, band_center, band_width, n_order):
    '''
        Butterworth Function
    '''
    bw_func = 1 / (1 + ((distance_map**2-band_center**2) / (distance_map*band_width))**(2*n_order))

    return bw_func

def bandpassFilter(filter_size, filter_pos, band_center, band_width, filter_func, n_order=2):
    '''
        Band-pass Filter in Frequency Domain
    '''
    # -> Distance Map 2D from given position 'filter_pos'
    distance_map = distanceMap(filter_size, filter_pos)

    # -> Create Frequency Filter from selected Function
    filterFunction = {
                        "Ideal": idealFunction(distance_map, band_center, band_width),
                        "Gaussian": gaussianFunction(distance_map, band_center, band_width),
                        "Butterworth": butterworthFunction(distance_map, band_center, band_width, n_order)
                     }
    bp_filer = filterFunction[filter_func]

    return bp_filer
    
