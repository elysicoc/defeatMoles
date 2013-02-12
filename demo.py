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

import skimage.io, skimage.color
from scipy import *
def fastSkinDetection(rgb, threshold=70, color=[255,20,147]):
	
    res = rgb.copy()
    lab = skimage.color.rgb2lab(rgb)
    mask = threshold< lab[:,:,0];	mask=ndimage.binary_erosion(mask,iterations=5); mask=ndimage.binary_dilation(mask,iterations=10); mask=invert(mask)
    res[mask] = array(color).reshape(1,-1).repeat(mask.sum(),axis=0)
    return res
	
def skinDetection(img, treshold=80, color=[255,20,147]):

	print img.shape
	res=img.copy()
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			RGBimg=RGBColor(img[x,y,0],img[x,y,1],img[x,y,2])
			LABimg=RGBimg.convert_to('lab', debug=False)
			if (LABimg.lab_l < treshold):
				res[x,y,:]=color
			else: 
				res[x,y,:]=img[x,y,:]

	return res
	
	

import cv2

def moleDetection(img, minArea=20,maxArea=800):
	params=cv2.SimpleBlobDetector_Params()
	params.filterByColor=True
	params.filterByConvexity=False
	params.filterByInertia=True
	params.minArea=minArea
	params.maxArea=maxArea	
	detector=cv2.SimpleBlobDetector(params)
	pts=detector.detect(img)

	return pts

def matching(b0,b1, thresholding=True, minArea=10, maxArea=100, thres_min=80, thres_max=300, ang_min=0, ang_max=0.5):

	moles0=moleDetection(b0, minArea, maxArea)
	moles1=moleDetection(b1, minArea, maxArea)
	
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

	
	# #####################################
	# visualization
	h1, w1 = b0.shape[:2]
	h2, w2 = b1.shape[:2]
	view = zeros((max(h1, h2), w1 + w2,3), uint8)
	view[:h1, :w1, 0] = b0[:,:,0]
	view[:h2, w1:, 0] = b1[:,:,0]
	view[:, :, 1] = view[:, :, 0]
	view[:, :, 2] = view[:, :, 0]


	
	if thresholding is True:

		# keep only the reasonable matches
		sel_matches=[]
		for m in matches:
			ang=(k1[m.queryIdx].pt[1] - k2[m.trainIdx].pt[1])/ (k1[m.queryIdx].pt[0] -k2[m.trainIdx].pt[0])  
			print ((m.distance < thres_max) and (m.distance>thres_min) and ang<ang_max and ang>ang_min), m.distance, ang
			if ((m.distance < thres_max) and (m.distance>thres_min) and ang<ang_max and ang>ang_min):
				sel_matches.append(m)
				
	else:
		sel_matches=matches

	print '#selected matches:', len(sel_matches)





	for m in sel_matches:
		# draw the keypoints
		#print m.queryIdx, m.trainIdx, m.distance
		color = tuple([random.randint(0, 255) for _ in xrange(3)]); 
		d_size=(k1[m.queryIdx].size)-(k2[m.trainIdx].size)
		ang=(k1[m.queryIdx].pt[1] - k2[m.trainIdx].pt[1])/ (k1[m.queryIdx].pt[0] -k2[m.trainIdx].pt[0])  
		print  "point=",k1[m.queryIdx].pt,"  angle=", ang
		print "color=",color, "  sizes =",k1[m.queryIdx].size," ",k2[m.trainIdx].size, " ",d_size
		
		#print k1[m.queryIdx].size
		if d_size>1.5:
			print "woooo", m.queryIdx, k1[m.queryIdx].size-k2[m.trainIdx].size
			cv2.circle(view, (int(k1[m.queryIdx].pt[0]),int(k1[m.queryIdx].pt[1])), 20, color)

		cv2.line(view, (int(k1[m.queryIdx].pt[0]),int(k1[m.queryIdx].pt[1])), (int(k2[m.trainIdx].pt[0] + w1), int(k2[m.trainIdx].pt[1])), color)

	cv2.imwrite("C:/Users/Kiarash/Dropbox/Git/defeatMoles/images/result.jpg",view)

	
b1= imread("C:/Users/Kiarash/Dropbox/Git/defeatMoles/images/black_1.jpg")
b0= imread("C:/Users/Kiarash/Dropbox/Git/defeatMoles/images/black_2.jpg")


#skinb0=fastSkinDetection(b0, 70)
#skinb1=fastSkinDetection(b1,70)

#moles=moleDetection(b0,50,300)
matching(b0,b1,True, minArea=50,maxArea=300, thres_min=0, thres_max=10000, ang_min=-1, ang_max=0)

#imsave("C:/Users/Kiarash/Dropbox/Git/defeatMoles/images/skin_black_2.jpg", skinb0)
#imsave("C:/Users/Kiarash/Dropbox/Git/defeatMoles/images/skin_black_1.jpg", skinb1)

'''
while(len(moles)>0):
	p=moles.pop().pt
	gca().add_patch(Circle(p, 50, facecolor='none', edgecolor='b'))	

imshow(b0)
'''

'''gray=rgb2gray(img).astype(uint8)
norm=normalize_histogram(gray)

coo=[1679,1350] #arm
#coo=[580,775] #smallarm2

match= templateMatching(norm, coo, 100,100)
 
imshow(match,cmap = plt.get_cmap('gray'))'''
show()

	
	
	