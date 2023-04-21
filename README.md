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

