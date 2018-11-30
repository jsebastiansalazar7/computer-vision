# USAGE
# In the command line type:
# python faceDetection.py --face Cascades/haarcascade_frontalface_default.xml --image Images/obama.png

# Import necessary packages

from FaceDetectorClass.faceDetectorClass import FaceDetector
import argparse
import cv2

# Initialize argument parser and parse arguments

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True, 
	help = "Path to where the face cascade resides")
ap.add_argument("-i", "--image", required = True,
	help = "Path to where the image file resides")
args = vars(ap.parse_args())

# Load the image and convert it to Grayscale

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Instance the FaceDetector class

fd = FaceDetector(args["face"])

# Detect number of faces 

faceRects = fd.detect(gray, scaleFactor = 1.32, minNeighbors = 5,
	minSize = (30, 30))
if ((len(faceRects)) == 1):
	print "I found %d face" % (len(faceRects))
elif ((len(faceRects)) > 1):
	print "I found %d faces" % (len(faceRects))
else:
	print "There are no faces in this image"

# Draw a rectangle around the detected faces

for (x, y, w, h) in faceRects:
	cv2.rectangle(image, (x, y), (x + w, y + h), 
		(0, 255, 0), 2)
cv2.imshow("Faces", image)
cv2.waitKey(0)
