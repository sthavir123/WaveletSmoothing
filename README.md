# WaveletSmoothing
GNR 602 Project :  Implement wavelet transform based image smoothing

## Requirements
* python3
* tkinter(for GUI)
```
sudo apt install python3-tk
```
### Libraries

1. numpy
```
pip install numpy
```
2 .pywt
```
pip install PyWavelets
```
3. cv2
```
pip install opencv-python
```
4. math

## Usage
1. Run the program using the command
```
python3 display1.py
```
2. Select input file
3. Give appropiate values for parameters asked by the popup windows
  - Enter Method from ['UniversalThreshold','BayesShrink','VisuShrink']
  - Enter Mode from ['soft','hard']
  - Enter wavelet form ['db4','db6']
  - Enter an integer value for wavelet decomposition levels.
  - Enter a float value for sigma (standard deviation) of noise.
4. Select output path and file name
5. Denoised image is displayed and gets stored at specified output path.

## Refrences
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       :DOI:`10.1093/biomet/81.3.425`
    .. [2] Chang, S. G., Yu, B., and Vetterli, M. (2000). Adaptive wavelet
        thresholding for image denoising and compression. IEEE Trans. on
        Image Proc., 9, 1532-1546
    .. [3] Sachin, Mr & Assistant, Ruikar. (2010). Image Denoising Using Wavelet Transform. 
