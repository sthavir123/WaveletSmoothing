
import cv2
import numpy as np

from wavelet_thresholding import _wavelet_threshold

def main(path,wavelet='db4',method='BayesShrink',levels=1,sigma=None,mode = 'soft'):
    print(path,wavelet,method,sigma,mode)
    # Load the noisy image
    img = cv2.imread(path)
    
    #resize the image so that dimensions are :
    #requuired due to pywt library issue for odd sized inputs during wavercn
    if img.shape[0]%2!=0 or img.shape[1]%2!=0:
        k = int(img.shape[0]/2)
        t = int(img.shape[1]/2)
        img = cv2.resize(img,[k*2,t*2])
    # Convert the image to YCbCr color space
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Split the channels
    y, cr, cb = cv2.split(img_ycrcb)
    # apply wavelet threshold on each channel seperately
    y_denoised  = _wavelet_threshold(image=y, wavelet=wavelet,method=method,wavelet_levels=levels,mode=mode,sigma=sigma)
    cr_denoised = _wavelet_threshold(image=cr,wavelet=wavelet,method=method,wavelet_levels=levels,mode=mode,sigma=sigma)
    cb_denoised = _wavelet_threshold(image=cb,wavelet=wavelet,method=method,wavelet_levels=levels,mode=mode,sigma=sigma)
    
    #recontruct the image in YCrcCb color space
    img_denoised_ycrcb = np.zeros_like(img)
    img_denoised_ycrcb[:,:,0] = y_denoised
    img_denoised_ycrcb[:,:,1] = cr_denoised
    img_denoised_ycrcb[:,:,2] = cb_denoised
    
    #convert back to RGB color space
    img_denoised = cv2.cvtColor(img_denoised_ycrcb, cv2.COLOR_YCrCb2BGR)
    print(cv2.PSNR(img_denoised,img))
    return img_denoised
#testing    
