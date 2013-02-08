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
	
	imshow(temp,cmap = plt.get_cmap('gray'))
	show()
	ans=img.copy()
	shapex,shapey=temp.shape
	
	'''for x in range(img.shape[0]-shapex):
		for y in range(img.shape[1]-shapey):
			ans[x,y]=sum(abs(img[x:x+shapex,y:y+shapey]-temp))
	
	'''
	floatimg=img*1.0
	print floatimg.shape, temp.shape
	
	ans=ndimage.convolve(floatimg, temp)
	m=numpy.mean(ans)
	ans=ans-m
	
	return ans
	
def rgb2gray(rgb):
    r, g, b = numpy.rollaxis(rgb[...,:3], axis = -1)
    return 0.299 * r + 0.587 * g + 0.114 * b
	
	
from colormath.color_objects import *
def skinDetection(img, treshold=80, color=[255,20,147]):

	print img.shape
	res=img.copy()
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			RGBimg=RGBColor(img[x,y,0],img[x,y,1],img[x,y,2])
			LABimg=RGBimg.convert_to('lab', debug=False)
			if (LABimg.lab_l > treshold):
				res[x,y,:]=color
			else: 
				res[x,y,:]=img[x,y,:]

	return res
	
	
import cv2
def moleDetection(img, minArea=80):
	params=cv2.SimpleBlobDetector_Params()
	params.filterByColor=False
	params.filterByConvexity=False
	params.filterByInertia=True
	params.minArea=minArea
	detector=cv2.SimpleBlobDetector(params)
	pts=detector.detect(img)

	return pts
	
	
img= imread("C:/Users/Kiarash/Dropbox/Git/defeatMoles/img012.jpg")

#skin=skinDetection(img, 55)

moles=moleDetection(img)

while(len(moles)>0):
	p=moles.pop().pt
	gca().add_patch(Circle(p, 50, facecolor='none', edgecolor='r'))	


imshow(img)


'''gray=rgb2gray(img).astype(uint8)
norm=normalize_histogram(gray)

coo=[1679,1350] #arm
#coo=[580,775] #smallarm2

match= templateMatching(norm, coo, 100,100)
 
imshow(match,cmap = plt.get_cmap('gray'))'''
show()

	
	
	