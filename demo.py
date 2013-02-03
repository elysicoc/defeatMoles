import numpy 
import ndimage
from pylab import *
from scipy import *

def normalize_histogram(image, cutoff_percent=0.01):
    assert image.dtype == uint8, "Assumes uint8 only"
    px_count = image.size

    hist = zeros(2**8)
    bc = bincount(image.flatten())
    hist[0:bc.size] = bc

    hist_cdf = 1.* hist.cumsum() / px_count
    
    lower_cut = 0
    upper_cut = 255
    
    # TODO: Deal with higher-bit images.  
    gray_span = range(2**8)
    for x in gray_span:
        if hist_cdf[x] < cutoff_percent:  lower_cut = x


    for x in gray_span[::-1]:
        if hist_cdf[x] > 1 - cutoff_percent:  upper_cut = x

    lower_cut = lower_cut + (1./2)
    upper_cut = upper_cut - (1./2)

    rescaled = image.clip(lower_cut, upper_cut)
    rescaled -= rescaled.min()
    rescaled /= 1. * rescaled.max()
    rescaled *= 255.
    
    return rescaled.clip(0, 255).round().astype(uint8)

def templateMatching(img, coo, offx, offy):
	
	temp=img[coo[0]-offx:coo[0]+offx, coo[1]-offy:coo[1]+offy]
	
	#imshow(temp,cmap = plt.get_cmap('gray'))
	#show()
	
	floatimg=img*1.0
	
	ans=ndimage.convolve(floatimg, temp)
	m=numpy.mean(ans)
	ans=ans-m
	
	return ans
	
def rgb2gray(rgb):
    r, g, b = numpy.rollaxis(rgb[...,:3], axis = -1)
    return 0.299 * r + 0.587 * g + 0.114 * b
	
	
img= imread("C:/Users/Kiarash/Dropbox/Git/defeatMoles/smallArm2.jpg")
gray=rgb2gray(img).astype(uint8)
norm=normalize_histogram(gray)

coo=[579,769]

match= templateMatching(norm, coo, 100,100) 
imshow(match,cmap = plt.get_cmap('gray'))
show()

	
	
	