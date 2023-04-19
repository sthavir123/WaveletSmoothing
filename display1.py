import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from p import main

# Create the Tkinter window
root = tk.Tk()

file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("All files", "*")])
print(file_path)
#---------------------------------code starts--------------------------------------
import numpy as np
import pywt
import cv2



#print(image[:,:,0].shape)
# Smooth the image using wavelet transform
smoothed_image = main(path=file_path,wavelet='db4',method='BayesShrink',levels=2,sigma=2)
# Display the original and smoothed images
#cv2.imshow('Original Image', image)
cv2.imwrite('result.png', smoothed_image)


#----------------------------------code ends ---------------------


# Set the window size
window_width = 2200
window_height = 1000
root.geometry(f"{window_width}x{window_height}")

# Load the images using PIL

image = Image.open("result.png")

# Resize the images while maintaining their aspect ratio to fit the window
image = image.resize((int(window_width/4), int(window_height)), Image.ANTIALIAS)


# Convert the images to Tkinter-compatible format
tk_image = ImageTk.PhotoImage(image)


# Create a Tkinter label with the images and pack them side by side in the window
label = tk.Label(root, image=tk_image)
label.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
label_name = tk.Label(label, text="Smoothed Image")
label_name.pack(side=tk.TOP)


# Run the Tkinter event loop
root.mainloop()
