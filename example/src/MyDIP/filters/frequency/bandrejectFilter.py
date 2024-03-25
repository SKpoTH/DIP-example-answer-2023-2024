import numpy as np

from MyDIP.filters.frequency import bandpassFilter

def bandrejectFilter(filter_size, filter_pos, band_center, band_width, filter_func, n_order=2):
    '''
        Band-reject Filter in Frequency Domain
    '''
    # -> Create High-pass filter from Low-pass filter
    br_filter = 1 - bandpassFilter(filter_size, filter_pos, band_center, band_width, filter_func, n_order)
    
    return br_filter