import numpy as np
import matplotlib.pyplot as plt

from MyDIP.fourier import Fourier2D
from MyDIP.filters.frequency import highpassFilter

def notchFreq(input_img, pos_list, freq_cutoff, filter_func, n_order=2):
    '''
        Notch Filtering in Frequency Domain
    '''
    ### -> Fast Fourier Transform 2D
    # - Forward FFT
    FFT = Fourier2D(input_img)
    FFT.fft()
    FFT.showMagnitude(ban_radius=15)
    fft_magnitude = FFT.getMagnitude()
    # -> Notch filtering output buffer
    notch_filter = np.ones_like(fft_magnitude)

    for filter_pos in pos_list:
        # -> Given Position
        hp_filter = highpassFilter(fft_magnitude.shape[:2], filter_pos, freq_cutoff, filter_func, n_order)
        # -> Mirror Position
        filter_mpos = (fft_magnitude.shape[0]-filter_pos[0], fft_magnitude.shape[1]-filter_pos[1])
        hp_mfilter = highpassFilter(fft_magnitude.shape[:2], filter_mpos, freq_cutoff, filter_func, n_order)
        # -> Merge Notch Filter
        notch_filter = notch_filter * hp_filter * hp_mfilter
    
    # -> Notch Filtering
    ifft_magnitude = fft_magnitude * notch_filter

    # ~> Display Notch Filter
    plt.imshow(notch_filter, cmap="gray")
    plt.show()

    # - Inverse FFT
    FFT.setMagnitude(ifft_magnitude)
    # FFT.showMagnitude(ban_radius=150)
    FFT.ifft()
    output_img = FFT.getOutputImg()

    return output_img