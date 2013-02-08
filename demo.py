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

def moleDetection(img, minArea=120,maxArea=5000):
	params=cv2.SimpleBlobDetector_Params()
	params.filterByColor=True
	params.filterByConvexity=False
	params.filterByInertia=True
	params.minArea=minArea
	params.maxArea=maxArea	
	detector=cv2.SimpleBlobDetector(params)
	pts=detector.detect(img)

	return pts

def matching(b0,b1, threshold=100):

	moles0=moleDetection(b0)
	moles1=moleDetection(b1)
	
	extractor=cv2.DescriptorExtractor_create("SIFT")
	k1,d0=extractor.compute(b0, moles0)
	k2,d1=extractor.compute(b1, moles1)

	matcher=cv2.DescriptorMatcher_create("FlannBased")
	matches=matcher.match(d0,d1)


	# visualize the matches
	print '#matches:', len(matches)
	dist = [m.distance for m in matches]

	print 'distance: min: %.3f' % min(dist)
	print 'distance: mean: %.3f' % (sum(dist) / len(dist))
	print 'distance: max: %.3f' % max(dist)


	# threshold: half the mean
	thres_dist = threshold

	# keep only the reasonable matches
	sel_matches = [m for m in matches if m.distance < thres_dist]

	print '#selected matches:', len(sel_matches)


	# #####################################
	# visualization
	h1, w1 = b0.shape[:2]
	h2, w2 = b1.shape[:2]
	view = zeros((max(h1, h2), w1 + w2, 3), uint8)
	view[:h1, :w1, 0] = b0[:,:,0]
	view[:h2, w1:, 0] = b1[:,:,0]
	view[:, :, 1] = view[:, :, 0]
	view[:, :, 2] = view[:, :, 0]


	for m in sel_matches:
		# draw the keypoints
		# print m.queryIdx, m.trainIdx, m.distance
		color = tuple([random.randint(0, 255) for _ in xrange(3)]); print  k1[m.queryIdx].pt
		cv2.line(view, (int(k1[m.queryIdx].pt[0]),int(k1[m.queryIdx].pt[1])), (int(k2[m.trainIdx].pt[0] + w1), int(k2[m.trainIdx].pt[1])), color)

		
		
	cv2.imshow("view", view)
	cv2.waitKey()

	
#b1= imread("C:/Users/Kiarash/Dropbox/Git/defeatMoles/back1.jpg")
b0= imread("C:/Users/Kiarash/Dropbox/Git/defeatMoles/img012.jpg")
skin=skinDetection(b0, 55)
moles=moleDetection(b0,80,100)
#matching(b0,b1)

while(len(moles)>0):
	p=moles.pop().pt
	gca().add_patch(Circle(p, 50, facecolor='none', edgecolor='r'))	

imshow(skin)

'''gray=rgb2gray(img).astype(uint8)
norm=normalize_histogram(gray)

coo=[1679,1350] #arm
#coo=[580,775] #smallarm2

match= templateMatching(norm, coo, 100,100)
 
imshow(match,cmap = plt.get_cmap('gray'))'''
show()

	
	
	