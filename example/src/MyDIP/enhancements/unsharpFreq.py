import numpy as np

from MyDIP.fourier import Fourier2D
from MyDIP.filters.frequency import lowpassFilter

def unsharpFreq(input_img, freq_cutoff, filter_func, n_order=2, k=2):
    '''
        Unsharp Masking in Frequency Domain
    '''
    ### -> Fast Fourier Transform 2D
    # - Forward FFT
    FFT = Fourier2D(input_img)
    FFT.fft()
    fft_magnitude = FFT.getMagnitude()

    # -> Create Low-pass Filter in Frequency Domain
    center_pos = (fft_magnitude.shape[0]//2, fft_magnitude.shape[1]//2)
    lp_filter = lowpassFilter(fft_magnitude.shape[:2], center_pos, freq_cutoff, filter_func, n_order)
    # -> Unsharp Masking & High Boosting
    ifft_magnitude = (1 + k * (1 - lp_filter)) * fft_magnitude

    # - Inverse FFT
    FFT.setMagnitude(ifft_magnitude)
    # FFT.showMagnitude()
    FFT.ifft()
    output_img = FFT.getOutputImg()

    ### -> Clip range into [0, 255]
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)

    return output_img