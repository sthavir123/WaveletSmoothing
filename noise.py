import numpy as np
import cv2

def add_speckle_noise(img, mean=0, var=0.1):
    row, col, ch = img.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    noisy = img + img * gauss * var + mean
    return noisy

# Load color image
img = cv2.imread('russia_org.jpg')
#img = cv2.resize(img,[480,480])
# Add speckle noise with mean=0 and variance=0.1\\
k = int(img.shape[0]/2)
t = int(img.shape[1]/2)
img = cv2.resize(img,[k*2,t*2])
noisy_img = add_speckle_noise(img, mean=0, var=0.4)

#noisy_img = cv2.resize(noisy_img,[k*2,t*2])

# Display original and noisy image side by side
#cv2.imshow('Original', img)
cv2.imwrite('russia_var0.5.jpg', noisy_img.astype(np.uint8))
cv2.imwrite('russia_org.jpg',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()