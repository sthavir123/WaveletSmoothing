import numpy as np
from scipy.signal import convolve2d
import cv2
def dwt2(image, wavelet='haar', mode='periodization'):
    """
    2D Discrete Wavelet Transform using only the dwt function.

    Args:
        image (ndarray): Input 2D image.
        wavelet (str, optional): Wavelet to use for the transformation.
            Default is 'haar'.
        mode (str, optional): Signal extension mode. Default is 'periodization'.

    Returns:
        (cA, (cH, cV, cD)): Tuple containing the approximation coefficients
        and the detail coefficients in horizontal, vertical, and diagonal
        directions, respectively.
    """
    # Decomposition filter and scaling filter for the selected wavelet
    if wavelet == 'haar':
        # Haar wavelet filters
        h0 = np.array([1, 1]) / np.sqrt(2)
        h1 = np.array([-1, 1]) / np.sqrt(2)
    elif wavelet == 'db1':
        # Daubechies-1 wavelet filters
        h0 = np.array([1, 1]) / np.sqrt(2)
        h1 = np.array([-1, 1]) / np.sqrt(2)
    else:
        # Unsupported wavelet
        raise ValueError('Unsupported wavelet: {}'.format(wavelet))

    # Convolve rows with the decomposition filter, then downsample
    cA = convolve2d(image, h0[np.newaxis, :], mode=mode, boundary='symm', axis=1)[:, ::2]
    cDh = convolve2d(image, h1[np.newaxis, :], mode=mode, boundary='symm', axis=1)[:, ::2]

    # Convolve columns of approximation with the decomposition filter, then downsample
    cA = convolve2d(cA, h0[:, np.newaxis], mode=mode, boundary='symm', axis=0)[::2, :]
    cDv = convolve2d(cA, h1[:, np.newaxis], mode=mode, boundary='symm', axis=0)[::2, :]

    # Convolve columns of detail with the decomposition filter, then downsample
    cDh = convolve2d(cDh, h0[:, np.newaxis], mode=mode, boundary='symm', axis=0)[::2, :]
    cDd = convolve2d(cDh, h1[:, np.newaxis], mode=mode, boundary='symm', axis=0)[::2, :]

    return (cA, (cDh, cDv, cDd))

img = cv2.imread('Figure_1.png')
print(dwt2(image=(img[:,:,0]+img[:,:,1]+img[:,:,2])/3));
#cv2.imshow('Reconstructed', k)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
