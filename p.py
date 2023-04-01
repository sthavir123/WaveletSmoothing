import numpy as np
import pywt
import cv2

def wavelet_image_smoothing(image, level=1, wavelet='db4'):
    # Convert the image to grayscale
    #if len(image.shape) == 3:
    #    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Apply the wavelet transform
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # Set the smallest coefficients to zero
    threshold = np.std(coeffs[-level]) * 10
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        new_coeffs.append(tuple([np.where(np.abs(coeffs[i][j]) < threshold, 0, coeffs[i][j]) for j in range(len(coeffs[i]))]))
    # Apply the inverse wavelet transform
    smoothed_image = pywt.waverec2(new_coeffs, wavelet)
    
    # Convert the image back to uint8 format
    smoothed_image = np.uint8(smoothed_image)
    
    return smoothed_image

# Read an image
image = cv2.imread('Figure_1.png')
image = cv2.resize(image,(500,500))
#print(image[:,:,0].shape)
# Smooth the image using wavelet transform
smoothed_image1 = wavelet_image_smoothing(image[:,:,0], level=1, wavelet='db6')
smoothed_image2 = wavelet_image_smoothing(image[:,:,1], level=1, wavelet='db6')
smoothed_image3 = wavelet_image_smoothing(image[:,:,2], level=1, wavelet='db6')
# Display the original and smoothed images
#cv2.imshow('Original Image', image)
cv2.imwrite('S1.png', smoothed_image1)
cv2.imwrite('S2.png', smoothed_image2)
cv2.imwrite('S3.png', smoothed_image3)

s_image = image
s_image[:,:,0] = smoothed_image1
s_image[:,:,1] = smoothed_image2
s_image[:,:,2] = smoothed_image3

cv2.imwrite('S.png', s_image)