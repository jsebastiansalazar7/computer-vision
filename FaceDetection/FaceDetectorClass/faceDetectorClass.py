# Import the necessary packages

import cv2

# Define the class to perform face detection

class FaceDetector:

# Constructor definition

	def __init__(self, faceCascadePath):
		self.faceCascade = cv2.CascadeClassifier(
			faceCascadePath)

# Method to find faces in images

	def detect(self, image, scaleFactor = 1.1, 
		minNeighbors = 5, minSize = (30,30)):
		rects = self.faceCascade.detectMultiScale(
			image, scaleFactor = scaleFactor, 
			minNeighbors = minNeighbors, 
			minSize = minSize, 
			flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
		return rects
