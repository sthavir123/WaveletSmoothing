import math
import cv2
import numpy as np
import pywt
from test import _wavelet_threshold
def threshold(coeffs,thresh):
    coeffs_thresh = list(coeffs)
    coeffs_thresh[0] = coeffs[0]
    for i in range(1, len(coeffs)):
        coeffs_thresh[i] = pywt.threshold(coeffs[i], thresh, mode='soft')
    return coeffs_thresh


def mse(img1, img2):
   h, w , c = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w*c))
   return mse

def denoise (path,params):
    # Load the noisy image
    img = cv2.imread(path)

    # Convert the image to YCbCr color space
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Split the channels
    y, cr, cb = cv2.split(img_ycrcb)

    # Apply wavelet transform on the Y channel
    coeffs = pywt.dwt2(y, params[0])
    #sigma = np.median(np.abs(pywt.wavedec2(img, 'db4', mode='symmetric', level=4)[1])) / 0.6745
    # Threshold the high-frequency coefficients
    
    coeffs_thresh = threshold(coeffs,params[1])
    #print(len(coeffs),len(coeffs_thresh))

    # Reconstruct the denoised Y channel
    y_denoised = pywt.idwt2(coeffs_thresh, params[0])
    print(y.shape,y_denoised.shape)
    # Merge the channels
    img_denoised_ycrcb = img
    img_denoised_ycrcb[:,:,0] = y_denoised
    img_denoised_ycrcb[:,:,1] = cr
    img_denoised_ycrcb[:,:,2] = cb

    # Convert the image back to RGB color space
    img_denoised = cv2.cvtColor(img_denoised_ycrcb, cv2.COLOR_YCrCb2BGR)
    return img_denoised
    # Display the denoised image


def main(path,wavelet,method,levels=1,sigma=None):
    # Load the noisy image
    img = cv2.imread(path)
    #img2 = cv2.imread("russia_org.jpg")

        # Convert the image to YCbCr color space
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        # Split the channels
    y, cr, cb = cv2.split(img_ycrcb)
    y_denoised  = _wavelet_threshold(image=y, wavelet=wavelet,method=method,wavelet_levels=levels)
    cr_denoised = _wavelet_threshold(image=cr,wavelet=wavelet,method=method,wavelet_levels=levels)
    cb_denoised = _wavelet_threshold(image=cb,wavelet=wavelet,method=method,wavelet_levels=levels)
    img_denoised_ycrcb = np.zeros_like(img)
    img_denoised_ycrcb[:,:,0] = y_denoised
    img_denoised_ycrcb[:,:,1] = cr_denoised
    img_denoised_ycrcb[:,:,2] = cb_denoised
    img_denoised = cv2.cvtColor(img_denoised_ycrcb, cv2.COLOR_YCrCb2BGR)
    return img_denoised
    #cv2.imshow('Denoised image', img_denoised.astype(np.uint8))

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
