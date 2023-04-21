from logging import warn
import numpy as np
import pywt
import scipy
import scipy.stats
def _bayes_thresh(details, var):
    """BayesShrink threshold for a zero-mean details coeff array."""
    # Equivalent to:  dvar = np.var(details) for 0-mean details array
    dvar = np.mean(details*details)
    eps = np.finfo(details.dtype).eps
    thresh = var / np.sqrt(max(dvar - var, eps))
    return thresh


def _universal_thresh(img, sigma):
    """ Universal threshold used by the VisuShrink method """
    return sigma*np.sqrt(2*np.log(img.size))

def _sigma_est_dwt(detail_coeffs, distribution='Gaussian'):
    
    """
    References
    ----------
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       :DOI:`10.1093/biomet/81.3.425`
    """
    # Consider regions with detail coefficients exactly zero to be masked out
    detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]
    denom = scipy.stats.norm.ppf(0.75)
    sigma = np.median(np.abs(detail_coeffs)) / denom
    return sigma




def _wavelet_threshold(image, wavelet, method=None, threshold=None,
                       sigma=None, mode='soft', wavelet_levels=None):
    
    
    wavelet = pywt.Wavelet(wavelet)
    # original_extent is used to workaround PyWavelets issue
    # odd-sized input results in an image with 1 extra sample after waverecn
    original_extent = tuple(slice(s) for s in image.shape)

    # Determine the number of wavelet decomposition levels
    if wavelet_levels is None:
        # Determine the maximum number of possible levels for image
        wavelet_levels = pywt.dwtn_max_level(image.shape, wavelet)
        
        # Skip coarsest wavelet scales.
        wavelet_levels = max(wavelet_levels - 3, 1)

    coeffs = pywt.wavedecn(image, wavelet=wavelet, level=wavelet_levels)
    # Detail coefficients at each decomposition level
    dcoeffs = coeffs[1:]
    
    if sigma is None:
        # Estimate the noise via the method in [2]_
        detail_coeffs = dcoeffs[-1]['d' * image.ndim]
        sigma = _sigma_est_dwt(detail_coeffs, distribution='Gaussian')

    if threshold is None:
        var = sigma**2
        
        if method == "BayesShrink":
            # The BayesShrink thresholds
            threshold = [{key: _bayes_thresh(level[key], var) for key in level}
                         for level in dcoeffs]
        elif method == "VisuShrink":
            # The VisuShrink thresholds
            detail_coeffs = dcoeffs[-1]['d' * image.ndim]
            sigma = _sigma_est_dwt(detail_coeffs, distribution='Gaussian')
            threshold = _universal_thresh(image, sigma)
        elif method == "UniversalThreshold":
            threshold = _universal_thresh(image,sigma)

    if np.isscalar(threshold):
        # A single threshold for all coefficient arrays
        denoised_detail = [{key: pywt.threshold(level[key],
                                                value=threshold,
                                                mode=mode) for key in level}
                           for level in dcoeffs]
    else:
        # Dict of unique threshold coefficients for each detail coeff. array
        denoised_detail = [{key: pywt.threshold(level[key],
                                                value=thresh[key],
                                                mode=mode) for key in level}
                           for thresh, level in zip(threshold, dcoeffs)]
    denoised_coeffs = [coeffs[0]] + denoised_detail
    print(method,mode,sigma,wavelet)
    return pywt.waverecn(denoised_coeffs, wavelet)[original_extent]

