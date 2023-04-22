# WaveletSmoothing
GNR 602 Assignment :  Implement wavelet transform based image smoothing

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

## Produce outputs
1. result_guassian_bs.png : 
  - path : ./noisy_images/russia_guassian.jpg
  - method : BayesShrink
  - mode : soft
  - wavelet : db4
  - levels : 3
  - sigma : 2.0

2. result_guassian_ut.png:
  - path : ./noisy_images/russia_guassian.jpg
  - method : UniversalThreshold
  - mode : soft
  - wavelet : db4
  - levels : 2
  - sigma : 2.0

3. result_guassian_vs.png:
  - path : ./noisy_images/russia_guassian.jpg
  - method : VisuShrink
  - mode : hard
  - wavelet : db4
  - levels : 2
  - sigma : None (click cancel in tinkter popup)

4. result_s&p_bs.png : 
  - path : ./noisy_images/russia_s&p.jpg
  - method : BayesShrink
  - mode : soft
  - wavelet : db4
  - levels : 3
  - sigma : 20.0

5. result_s&p_ut.png:
  - path : ./noisy_images/russia_s&p.jpg
  - method : UniversalThreshold
  - mode : soft
  - wavelet : db4
  - levels : 3
  - sigma : 10.0

6. result_s&p_vs.png:
  - path : ./noisy_images/russia_s&p.jpg
  - method : VisuShrink
  - mode : hard
  - wavelet : db4
  - levels : 4
  - sigma : None (click cancel in tinkter popup)

## Notes
* The file noise.py was just used to generate noisy images from original images
* display1.py implements the tinkter GUI 
* main.py converts the image to YCrCb and calls wavelet_thresholding function. It then reconstruct YCrCb image and converts it back to RGB.
* wavelet_thresholding.py implements the actual algorithm for wavelet denoising. 

## Refrences
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       :DOI:`10.1093/biomet/81.3.425`
    .. [2] Chang, S. G., Yu, B., and Vetterli, M. (2000). Adaptive wavelet
        thresholding for image denoising and compression. IEEE Trans. on
        Image Proc., 9, 1532-1546
    .. [3] Sachin, Mr & Assistant, Ruikar. (2010). Image Denoising Using Wavelet Transform. 
