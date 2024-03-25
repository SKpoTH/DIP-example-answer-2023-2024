import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import fftpack

from MyDIP.intensityTransform import logTransform

class Fourier2D:
    def __init__(self, input_img):
        '''
            Constructor
        '''
        self.__input_img = input_img

    # <-------------------------> Transformation Functions <------------------------>
    def fft(self):
        '''
            Foward Fourier Transform
        '''
        # -> Fast Fourier Transform 
        fft_complex = fftpack.fft2(self.__input_img)
        # -> Split Magnitude Phase, consequently
        self.__fft_magnitude = np.abs(fft_complex)
        self.__fft_phase = np.arctan2(fft_complex.imag, fft_complex.real)
        # -> Shift Quadrant
        self.__fft_magnitude = fftpack.fftshift(self.__fft_magnitude)

    def ifft(self):
        '''
            Inverse Fast Fourier Transform
        '''
        # - Invert Shift Magnitude
        ifft_magnitude = fftpack.ifftshift(self.__fft_magnitude)
        # - Combine Magnitude Phase
        ifft_real = ifft_magnitude * np.cos(self.__fft_phase) 
        ifft_imag = ifft_magnitude * np.sin(self.__fft_phase)
        # - Combine into Complex
        ifft_complex = ifft_real + (ifft_imag * 1j)
        # -> Invert FFT
        output_complex = fftpack.ifft2(ifft_complex)
        # -> Get Image Data from Real part
        self.__output_img = output_complex.real
    
    # <-------------------------------> Visualization <----------------------------->
    def showMagnitude(self, ban_radius=3, save=False):
        '''
            Show Magnitude Visualization
        '''
        ### - Banning Circle
        center_ban = np.ones_like(self.__fft_magnitude)
        center_ban = np.ascontiguousarray(center_ban)
        # - Center Position
        center = (int(center_ban.shape[1]//2), int(center_ban.shape[0]//2))
        # - Draw Circle
        center_ban = cv.circle(center_ban, center, ban_radius, 0, -1)
        # -> Center Banning
        v_magnitude = self.__fft_magnitude * center_ban

        # center_x = int(center_ban.shape[1]//2)
        # center_y = int(center_ban.shape[0]//2)
        # rad_size = 5
        # v_magnitude[:, center_x-rad_size:center_x+rad_size+1] = 0
        # v_magnitude[center_y-rad_size:center_y+rad_size+1, :] = 0

        # -> Log Intensity Transform
        v_magnitude = v_magnitude / v_magnitude.max()
        v_magnitude = logTransform(v_magnitude, c=1, to_uint8=False)

        # ~> Display Magnitude
        plt.imshow(v_magnitude, cmap="hot")
        if not save:
            plt.show()

    def saveMagnitude(self, output_path, output_filename):
        '''
            Save Magnitude Visualization
        '''
        self.showMagnitude(ban_radius=3, save=True)
        plt.savefig(output_path + output_filename, dpi=500)

    # <-------------------------------> API Functions <----------------------------->
    def getOutputImg(self):
        '''
            Give 'output_img' value
        '''
        return self.__output_img

    def getMagnitude(self):
        '''
            Give 'fft_magnitude' value
        '''
        return self.__fft_magnitude

    def setMagnitude(self, fft_magnitude):
        '''
            Assign 'fft_magnitude' value
        '''
        self.__fft_magnitude = fft_magnitude