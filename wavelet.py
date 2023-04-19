"""Discrete Wavelet and Inverse Transform implementation"""

from wavelets import getWaveletDefinition
import cv2

class WaveletTransform:
    
    def __init__(self, waveletName):
        self.__wavelet__ = getWaveletDefinition(waveletName)

    def dwt(self, arrTime, level):
        
        arrHilbert = [0.] * level
        # shrinking value 8 -> 4 -> 2
        a = level >> 1

        for i in range(a):
            for j in range(self.__wavelet__.__motherWaveletLength__):
                k = (i << 1) + j

                # circulate the array if scale is higher
                while k >= level:
                    k -= level

                # approx & detail coefficient
                arrHilbert[i] += arrTime[k] * self.__wavelet__.decompositionLowFilter[j]
                arrHilbert[i + a] += arrTime[k] * self.__wavelet__.decompositionHighFilter[j]

        return arrHilbert

    def idwt(self, arrHilbert, level):
        
        arrTime = [0.] * level
        # shrinking value 8 -> 4 -> 2
        a = level >> 1

        for i in range(a):
            for j in range(self.__wavelet__.__motherWaveletLength__):
                k = (i << 1) + j

                # circulating the array if scale is higher
                while k >= level:
                    k -= level

                # summing the approx & detail coefficient
                arrTime[k] += (arrHilbert[i] * self.__wavelet__.reconstructionLowFilter[j] +
                               arrHilbert[i + a] * self.__wavelet__.reconstructionHighFilter[j])

        return arrTime

"""Base Transform for doing basic calls for dwt & idwt based on the dimensions called"""

import numpy as np
from math import log, pow
def getExponent(value):
    """Returns the exponent for the data Ex: 8 -> 3 [2 ^ 3]"""
    return int(log(value) / log(2.))
def scalb(f, scaleFactor):
    """Return the scale for the factor"""
    return f * pow(2., scaleFactor)

def isPowerOf2(number):
    """Checks if the length is equal to the power of 2"""
    power = getExponent(number)
    result = scalb(1., power)

    return result == number


class BaseTransform:
    """
    Transform class to call the Discrete Wavelet Transform on select wavelet based on
    the dimensions of the data

    Attributes
    ----------
    __wavelet: WaveletTransform
        object of the Wavelet class based on the wavelet name
    """

    def __init__(self, waveletName):
        self.__wavelet = WaveletTransform(waveletName)

    def getWaveletDefinition(self):
        """
        Returns the wavelet definition for the select wavelet

        Returns
        -------
        object
            object of the selected wavelet class
        """
        return self.__wavelet.__wavelet__

    

    def waveDec1(self, arrTime, level):
        """
        Single Dimension wavelet decomposition based on the levels

        Parameters
        ----------
        arrTime : array_like
            input array signal in Time domain
        level : int
            level for the decomposition power of 2

        Returns
        -------
        array_like
            coefficients Frequency or the Hilbert domain
        """
        length = 0
        arrHilbert = arrTime.copy()
        dataLength = len(arrHilbert)
        transformWaveletLength = self.__wavelet.__wavelet__.__transformWaveletLength__

        while dataLength >= transformWaveletLength and length < level:
            arrTemp = self.__wavelet.dwt(arrHilbert, dataLength)
            arrHilbert[: len(arrTemp)] = arrTemp

            dataLength >>= 1
            length += 1

        return arrHilbert

    def waveRec1(self, arrHilbert, level):
        """
        Single Dimension wavelet reconstruction based on the levels

        Parameters
        ----------
        arrHilbert : array_like
            input array signal in Frequency or the Hilbert domain
        level : int
            level for the decomposition power of 2

        Returns
        -------
        array_like
            coefficients Time domain
        """
        arrTime = arrHilbert.copy()
        dataLength = len(arrTime)
        transformWaveletLength = self.__wavelet.__wavelet__.__transformWaveletLength__
        h = transformWaveletLength

        steps = getExponent(dataLength)
        for _ in range(level, steps):
            h <<= 1

        while len(arrTime) >= h >= transformWaveletLength:
            arrTemp = self.__wavelet.idwt(arrTime, h)
            arrTime[: len(arrTemp)] = arrTemp

            h <<= 1

        return arrTime

    def waveDec2(self, matTime,level):
        """
        Two Dimension Multi-level wavelet decomposition based on the levels

        Parameters
        ----------
        matTime : array_like
            input matrix signal in Time domain

        Returns
        -------
        array_like
            coefficients Time domain
        """
        # shape
        noOfRows = len(matTime)
        noOfCols = len(matTime[0])

        #if not isPowerOf2(noOfRows) or not isPowerOf2(noOfCols):
        #    raise WrongLengthsOfData(WrongLengthsOfData.__cause__)

        # get the levels
        levelM = getExponent(noOfRows)
        levelN = getExponent(noOfCols)

        matHilbert = np.zeros(shape=(noOfRows, noOfCols))

        # rows
        for i in range(noOfRows):
            # run the decomposition on the row
            matHilbert[i] = self.waveDec1(matTime[i], level)

        # cols
        for j in range(noOfCols):
            # run the decomposition on the col
            matHilbert[:, j] = self.waveDec1(matHilbert[:, j], level)

        return matHilbert

    def waveRec2(self, matHilbert,level):
        """
        Two Dimension Multi-level wavelet reconstruction based on the levels

        Parameters
        ----------
        matHilbert : array_like
            input matrix signal in Frequency or the Hilbert domain

        Returns
        -------
        array_like
            coefficients Time domain
        """
        noOfRows = len(matHilbert)
        noOfCols = len(matHilbert[0])

        #if not isPowerOf2(noOfRows) or not isPowerOf2(noOfCols):
        #    raise WrongLengthsOfData(WrongLengthsOfData.__cause__)

        # getting the levels
        levelM = getExponent(noOfRows)
        levelN = getExponent(noOfCols)

        matHilbert = np.array(matHilbert)
        matTime = np.zeros(shape=(noOfRows, noOfCols))

        # rows
        for j in range(noOfCols):
            # run the reconstruction on the row
            matTime[:, j] = self.waveRec1(matHilbert[:, j], level)

        # cols
        for i in range(noOfRows):
            # run the reconstruction on the column
            matTime[i] = self.waveRec1(matTime[i], level)

        return matTime

img = cv2.imread('Figure_1.png')


img = cv2.resize(img,(512,512))
img2 = np.zeros(img.shape)
t = BaseTransform('db4')
a1 = t.waveDec2(matTime=(img[:,:,0]),level=1)
a2 = t.waveDec2(matTime=(img[:,:,1]),level=1)
a3 = t.waveDec2(matTime=(img[:,:,2]),level=1)
k1 = t.waveRec2(matHilbert=a1,level=1)
k2 = t.waveRec2(matHilbert=a2,level=1)
k3 = t.waveRec2(matHilbert=a3,level=1)

img2[:,:,0] = k1
img2[:,:,1] = k2
img2[:,:,2] = k3



cv2.imshow('Reconstructed', k1.astype(np.uint8))
cv2.waitKey(0)
cv2.imshow('Reconstructed', k2.astype(np.uint8))
cv2.waitKey(0)
cv2.imshow('Reconstructed', img2)

#cv2.imshow('Reconstructed', img[:,:,0])

cv2.waitKey(0)
cv2.destroyAllWindows()
