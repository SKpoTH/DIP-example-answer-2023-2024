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

def idealFunction(distance_map, freq_cutoff):
    '''
        Ideal Function
    '''
    ideal_func = distance_map.copy()
    ideal_func[distance_map <= freq_cutoff] = 1
    ideal_func[distance_map > freq_cutoff] = 0

    return ideal_func

def gaussianFunction(distance_map, freq_cutoff):
    '''
        Gaussian Function
    '''
    gauss_func = np.exp(-distance_map**2 / (2*freq_cutoff**2))

    return gauss_func

def butterworthFunction(distance_map, freq_cutoff, n_order):
    '''
        Butterworth Function
    '''
    bw_func = 1 / (1 + (distance_map/freq_cutoff)**(2*n_order))

    return bw_func

def lowpassFilter(filter_size, filter_pos, freq_cutoff, filter_func, n_order=2):
    '''
        Low-pass Filter in Frequency Domain
    '''
    # -> Distance Map 2D from given position 'filter_pos'
    distance_map = distanceMap(filter_size, filter_pos)

    # -> Create Frequency Filter from selected Function
    filterFunction = {
                        "Ideal": idealFunction(distance_map, freq_cutoff),
                        "Gaussian": gaussianFunction(distance_map, freq_cutoff),
                        "Butterworth": butterworthFunction(distance_map, freq_cutoff, n_order)
                     }
    lp_filer = filterFunction[filter_func]

    return lp_filer
    
