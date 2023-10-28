import numpy as np
import pywt
from numpy import clip
from statsmodels.robust import mad


def waveletSmooth( x, wavelet="db6", level=1):
    output = []
    data = x.squeeze()

    for sample in data:
        # calculate the wavelet coefficients
        coeff = pywt.wavedec(sample, wavelet, mode="per", level=level)
        # calculate a threshold
        sigma = mad( coeff[-level] )
        # changing this threshold also changes the behavior,
        # but I have not played with this very much
        uthresh = sigma * np.sqrt( 2*np.log( len( sample ) ) )
        coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )
        # reconstruct the signal using the thresholded coefficients
        y = pywt.waverec( coeff, wavelet, mode="per" )
        y = clip(y,-1,1)
        output.append(y)
    return np.array(output)

