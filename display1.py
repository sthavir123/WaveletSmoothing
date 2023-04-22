import tkinter as tk
from tkinter import filedialog,simpledialog
from PIL import ImageTk, Image
from main import main

# Create the Tkinter window
root = tk.Tk()
window_height = 800
window_width = 800
root.geometry(f"{window_width}x{window_height}")
file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("All files", "*")])
print(file_path)


method = simpledialog.askstring("Input", "Enter Method:", parent=root)
mode = simpledialog.askstring("Input","Enter thresholding mode",parent=root)
wavelet = simpledialog.askstring("Input", "Enter wavelet:", parent=root)
levels = simpledialog.askinteger("Input", "Enter levels:", parent=root, minvalue=1)
sigma = simpledialog.askfloat("Input", "Enter sigma:", parent=root, minvalue=0.0)

# Print the values for testing
print("Method:",method)
print("Mode",mode)
print("Wavelet:", wavelet)
print("Levels:", levels)
print("Sigma:", sigma)


output_path = filedialog.asksaveasfilename(title="Save Output Image", filetypes=[("All files", "*")])

print(output_path)
#---------------------------------code starts--------------------------------------
import numpy as np
import pywt
import cv2



#print(image[:,:,0].shape)
# Smooth the image using wavelet transform
smoothed_image = main(path=file_path,wavelet=wavelet,method=method,levels=levels,sigma=sigma,mode=mode)
# Display the original and smoothed images
#cv2.imshow('Original Image', image)
cv2.imwrite(output_path, smoothed_image)


#----------------------------------code ends ---------------------


# Set the window size
window_width = smoothed_image.shape[1]
window_height = smoothed_image.shape[0]
root.geometry(f"{window_width}x{window_height}")

# Load the images using PIL

image = Image.open(output_path)

# Resize the images while maintaining their aspect ratio to fit the window
image = image.resize((int(window_width), int(window_height)), Image.ANTIALIAS)


# Convert the images to Tkinter-compatible format
tk_image = ImageTk.PhotoImage(image)


# Create a Tkinter label with the images and pack them side by side in the window
label = tk.Label(root, image=tk_image)
label.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
label_name = tk.Label(label, text="Smoothed Image")
label_name.pack(side=tk.TOP)


# Run the Tkinter event loop
root.mainloop()
