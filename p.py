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

def universal_threshold(coeffs,sigma):
    
    coeffs_thresh = list(coeffs)
    thresh = sigma*math.sqrt(2*math.log(len(coeffs[0])))
    print(thresh)
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

def nlb_threshold(coeffs, sigma, h, p):
    """
    Non-local Bayes thresholding function for wavelet coefficients
    coeffs: list of wavelet coefficients
    sigma: estimated noise variance
    h: non-local similarity parameter
    p: patch size
    """
    coeffs_thresh = list(coeffs)
    coeffs_thresh[0] = coeffs[0]
    for i in range(1, len(coeffs)):
        c = coeffs[i]
        c_thresh = np.zeros_like(c)
        for j in range(c.shape[0]):
            for k in range(c.shape[1]):
                patch = c[max(0, j-p):min(c.shape[0], j+p+1), max(0, k-p):min(c.shape[1], k+p+1)]
                patch_dists = np.sum((patch - c[j,k])**2, axis=(2,3))
                patch_weights = np.exp(-patch_dists / (h**2 * sigma))
                patch_weights /= np.sum(patch_weights)
                c_thresh[j,k] = np.sum(patch_weights * patch)
        coeffs_thresh[i] = c_thresh
    return coeffs_thresh



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

img = cv2.imread('russia_var0.5.jpg')
# Load the noisy image


    # Convert the image to YCbCr color space
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Split the channels
y, cr, cb = cv2.split(img_ycrcb)
y_denoised = _wavelet_threshold(image=y,wavelet='db8',method='VisuShrink',sigma=2,wavelet_levels=3)
cr_denoised = _wavelet_threshold(image=cr,wavelet='db8',method='VisuShrink',sigma=2,wavelet_levels=3)
cb_denoised = _wavelet_threshold(image=cb,wavelet='db8',method='VisuShrink',sigma=2,wavelet_levels=3)
img_denoised_ycrcb = img
img_denoised_ycrcb[:,:,0] = y_denoised
img_denoised_ycrcb[:,:,1] = cr
img_denoised_ycrcb[:,:,2] = cb
img_denoised = cv2.cvtColor(img_denoised_ycrcb, cv2.COLOR_YCrCb2BGR)


# applying kernels to the input image to get the sharpened image
img2 = cv2.imread("russia_org.jpg")

print(mse(img2,img_denoised))
print(mse(img2,img))
cv2.imshow('Denoised image', img_denoised.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()
