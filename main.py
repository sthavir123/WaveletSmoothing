
import cv2
import numpy as np

from wavelet_thresholding import _wavelet_threshold

def MSE(img1, img2):
        squared_diff = (img1 -img2) ** 2
        summed = np.sum(squared_diff)
        num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
        err = summed / num_pix
        return np.sqrt(err)

def main(path,wavelet='db4',method='BayesShrink',levels=1,sigma=None,mode = 'soft'):
    print(path,wavelet,method,sigma,mode)
    # Load the noisy image
    img = cv2.imread(path)
    #img2 = cv2.imread('./sample_images/russia_org.jpg')
    
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
    
    #print("MSE denoise : ",MSE(img_denoised,img2))
    #print("MSE noise   : ",MSE(img,img2))
    return img_denoised
#testing    
# imgden = main(path = 'noisy_images/russia_s&p.jpg'
#   ,method = 'VisuShrink'
#   ,mode = 'soft'
#   ,wavelet = 'db4'
#   ,levels = 2
#   ,sigma = None)
# cv2.imwrite('output_images/result_s&p_vs.png',imgden)
