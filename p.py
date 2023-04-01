import numpy as np
import pywt
import cv2

def wavelet_image_smoothing(image, level=1, wavelet='db4'):
    # Convert the image to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Apply the wavelet transform
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # Set the smallest coefficients to zero
    threshold = np.std(coeffs[-level]) * 3
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

# Smooth the image using wavelet transform
smoothed_image = wavelet_image_smoothing(image, level=1, wavelet='db4')

# Display the original and smoothed images
cv2.imshow('Original Image', image)
cv2.imshow('Smoothed Image', smoothed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()