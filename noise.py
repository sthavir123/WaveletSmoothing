
# code to generate noisy images
import numpy as np
import cv2
from skimage.util import random_noise


def add_speckle_noise(img, mean=0, var=0.1):
    row, col, ch = img.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    noisy = img + img * gauss * var + mean
    return noisy

img = cv2.imread('./sample_images/russia_org.jpg')

# Add speckle noise with mean=0 and variance=0.1\\

noisy_img = add_speckle_noise(img, mean=0, var=0.2)

noise_img = random_noise(img, mode='s&p')
noise_img = (255*noise_img).astype(np.uint8)


cv2.imwrite('./noisy_images/russia_s&p.jpg', noise_img)

