import cv2
import numpy as np
import pywt

def threshold(coeffs,thresh):
    coeffs_thresh = list(coeffs)
    coeffs_thresh[0] = coeffs[0]
    for i in range(1, len(coeffs)):
        coeffs_thresh[i] = pywt.threshold(coeffs[i], thresh, mode='soft')
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

    # Threshold the high-frequency coefficients
    coeffs_thresh = threshold(coeffs,params[1])
    

    # Reconstruct the denoised Y channel
    y_denoised = pywt.idwt2(coeffs_thresh, params[0])

    # Merge the channels
    img_denoised_ycrcb = img
    img_denoised_ycrcb[:,:,0] = y_denoised
    img_denoised_ycrcb[:,:,1] = cr
    img_denoised_ycrcb[:,:,2] = cb

    # Convert the image back to RGB color space
    img_denoised = cv2.cvtColor(img_denoised_ycrcb, cv2.COLOR_YCrCb2BGR)
    return img_denoised
    # Display the denoised image
cv2.imshow('Denoised image', denoise('Figure_1.png',['db4',20]))
cv2.waitKey(0)
cv2.destroyAllWindows()
