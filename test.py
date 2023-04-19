from logging import warn
import numpy as np
import pywt
import scipy

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

    if distribution.lower() == 'gaussian':
        # 75th quantile of the underlying, symmetric noise distribution
        denom = scipy.stats.norm.ppf(0.75)
        sigma = np.median(np.abs(detail_coeffs)) / denom
    else:
        raise ValueError("Only Gaussian noise estimation is currently "
                         "supported")
    return sigma




def _wavelet_threshold(image, wavelet, method=None, threshold=None,
                       sigma=None, mode='soft', wavelet_levels=None):
    
    
    wavelet = pywt.Wavelet(wavelet)
    if not wavelet.orthogonal:
        warn(f'Wavelet thresholding was designed for '
             f'use with orthogonal wavelets. For nonorthogonal '
             f'wavelets such as {wavelet.name},results are '
             f'likely to be suboptimal.')

    # original_extent is used to workaround PyWavelets issue #80
    # odd-sized input results in an image with 1 extra sample after waverecn
    original_extent = tuple(slice(s) for s in image.shape)

    # Determine the number of wavelet decomposition levels
    if wavelet_levels is None:
        # Determine the maximum number of possible levels for image
        wavelet_levels = pywt.dwtn_max_level(image.shape, wavelet)
        print(image.shape)
        # Skip coarsest wavelet scales (see Notes in docstring).
        wavelet_levels = max(wavelet_levels - 3, 1)

    coeffs = pywt.wavedecn(image, wavelet=wavelet, level=wavelet_levels)
    # Detail coefficients at each decomposition level
    dcoeffs = coeffs[1:]

    if sigma is None:
        # Estimate the noise via the method in [2]_
        detail_coeffs = dcoeffs[-1]['d' * image.ndim]
        sigma = _sigma_est_dwt(detail_coeffs, distribution='Gaussian')

    if method is not None and threshold is not None:
        warn(f'Thresholding method {method} selected. The '
             f'user-specified threshold will be ignored.')

    if threshold is None:
        var = sigma**2
        if method is None:
            raise ValueError(
                "If method is None, a threshold must be provided.")
        elif method == "BayesShrink":
            # The BayesShrink thresholds from [1]_ in docstring
            threshold = [{key: _bayes_thresh(level[key], var) for key in level}
                         for level in dcoeffs]
        elif method == "VisuShrink":
            # The VisuShrink thresholds from [2]_ in docstring
            threshold = _universal_thresh(image, sigma)
        else:
            raise ValueError(f'Unrecognized method: {method}')

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
    return pywt.waverecn(denoised_coeffs, wavelet)[original_extent]

def denoise_wavelet(image, sigma=None, wavelet='db1', mode='soft',
                    wavelet_levels=None,
                    convert2ycbcr=True, method='BayesShrink',
                    rescale_sigma=True, *, channel_axis=None):
    """
    
    References
    ----------
    .. [1] Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet
           thresholding for image denoising and compression." Image Processing,
           IEEE Transactions on 9.9 (2000): 1532-1546.
           :DOI:`10.1109/83.862633`
    .. [2] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
           by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
           :DOI:`10.1093/biomet/81.3.425`
    
    """
    multichannel = channel_axis is not None
    if method not in ["BayesShrink", "VisuShrink"]:
        raise ValueError(f'Invalid method: {method}. The currently supported '
                         f'methods are "BayesShrink" and "VisuShrink".')

    # floating-point inputs are not rescaled, so don't clip their output.
    clip_output = image.dtype.kind != 'f'

    if convert2ycbcr and not multichannel:
        raise ValueError("convert2ycbcr requires channel_axis to be set")

    image, sigma = _scale_sigma_and_image_consistently(image,
                                                       sigma,
                                                       multichannel,
                                                       rescale_sigma)

    if convert2ycbcr:
        out = color.rgb2ycbcr(image)
        # convert user-supplied sigmas to the new colorspace as well
        if rescale_sigma:
            sigma = _rescale_sigma_rgb2ycbcr(sigma)
        for i in range(3):
            # renormalizing this color channel to live in [0, 1]
            _min, _max = out[..., i].min(), out[..., i].max()
            scale_factor = _max - _min
            if scale_factor == 0:
                # skip any channel containing only zeros!
                continue
            channel = out[..., i] - _min
            channel /= scale_factor
            sigma_channel = sigma[i]
            if sigma_channel is not None:
                sigma_channel /= scale_factor
            out[..., i] = denoise_wavelet(channel,
                                          wavelet=wavelet,
                                          method=method,
                                          sigma=sigma_channel,
                                          mode=mode,
                                          wavelet_levels=wavelet_levels,
                                          rescale_sigma=rescale_sigma)
            out[..., i] = out[..., i] * scale_factor
            out[..., i] += _min
        out = color.ycbcr2rgb(out)
    


    if clip_output:
        clip_range = (-1, 1) if image.min() < 0 else (0, 1)
        out = np.clip(out, *clip_range, out=out)
    return out

