# USAGE
# In the command line type:
# 
# Using a video file:
# python webCamFaceDetection.py --face Cascades/haarcascade_frontalface_default.xml --video Videos/Me.mov
#
# Using a webcam:
# python webCamFaceDetection.py --face Cascades/haarcascade_frontalface_default.xml

# Import the necessary packages

from FaceDetectorClass.faceDetectorClass import FaceDetector
from FaceDetectorClass import imutils
import argparse
import cv2

# Initialize argument parser & parse arguments

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True, help = "Path to where the face cascade resides")
ap.add_argument("-v", "--video", help = "Path to where the (optional) video file")
args = vars(ap.parse_args())

# Instance the FaceDetector Class

fd = FaceDetector(args["face"])

# Read from webcam or video file

if not args.get("video", False):
	camera = cv2.VideoCapture(0)
else:
	camera = cv2.VideoCapture(args["video"])

# Read frames until video ends or is stoped

while True:

	(grabbed, frame) = camera.read()

	if args.get("video") and not grabbed:
		break

	# Set the frame's size and convert to grayscale

	frame = imutils.resize(frame, width = 300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Handle detection using the detect method of the FaceDetector class

	faceRects = fd.detect(gray, scaleFactor = 1.1, 
		minNeighbors = 5, minSize = (30,30))
	frameClone = frame.copy()

	for (fX, fY, fW, fH) in faceRects:
		
		cv2.rectangle(frameClone, (fX, fY), 
			(fX+fW, fY+fH), (0, 255, 0), 2)
	
	cv2.imshow("Face Detection", frameClone)

	# Break if user press "q" (Force quit with ctl+c)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		
		break

camera.release()
cv2.destroyAllWindows()
