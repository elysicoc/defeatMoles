from PIL import Image
from PIL import ImageChops
import numpy

def matchTemplate(searchImage, templateImage):
	
	searchImage = searchImage.convert(mode="L")
	templateImage = templateImage.convert(mode="L")
	searchWidth, searchHeight = searchImage.size
	templateWidth, templateHeight = templateImage.size
	templateMask = Image.new(mode="L", size=templateImage.size, color=1)
	score= numpy.zeros(1680*948).reshape(1680, 948)
	
	for xs in range(searchWidth-templateWidth+1):
		for ys in range(searchHeight-templateHeight+1):

			searchCrop = searchImage.crop((xs,ys,xs+templateWidth,ys+templateHeight))
			diff = ImageChops.difference(templateImage, searchCrop)
			notequal = ImageChops.darker(diff,templateMask)
			countnotequal = numpy.sum(notequal.getdata())
			score[xs,ys] = countnotequal

	print score


	score.save("C:/Users/Kiarash/Dropbox/Git/defeatMoles/ans2.jpg")

searchImage = Image.open("C:/Users/Kiarash/Dropbox/Git/defeatMoles/smallArm.jpg")
templateImage = Image.open("C:/Users/Kiarash/Dropbox/Git/defeatMoles/sample.jpg")

matchTemplate(searchImage, templateImage)
