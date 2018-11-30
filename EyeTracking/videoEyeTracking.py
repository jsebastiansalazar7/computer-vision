# USAGE
# In the command line type:
#
# Using a video file:
# python videoEyeTracking.py --face Cascades/haarcascade_frontalface_default.xml --eye Cascades/haarcascade_eye.xml --video Videos/MisaAndValen.mov
#
# Using a webcam:
# python webCamFaceDetection.py --face Cascades/haarcascade_frontalface_default.xml  --eye Cascades/haarcascade_eye.xml

# Import the necessary packages

from EyeDetectorClass.eyeDetectorClass import EyeTracker
from EyeDetectorClass import imutils
import argparse
import cv2

# Initialize argument parser & parse arguments

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True, 
	help = "Path to where the face cascade resides")
ap.add_argument("-e", "--eye", required = True,
	help = "Path to where the eye cascade resides")
ap.add_argument("-v", "--video", 
	help = "Path to the (optional) video file")
args = vars(ap.parse_args())

# Instance the EyeTracker class:

et = EyeTracker(args["face"], args["eye"])

# Read from webcam or video file

if not args.get("video", False):
	camera = cv2.VideoCapture(0)
else:
	camera = cv2.VideoCapture(args["video"])

# Read frames until video ends or is stopped

while True:

	(grabbed, frame) = camera.read()

	if args.get("video") and not grabbed:
		break

	# Set the frame's size and convert to grayscale

	frame = imutils.resize(frame, width = 300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Handle detection using the track method of the EyeTracker class

	rects = et.track(gray)

	for rect in rects:

		cv2.rectangle(frame, (rect[0], rect[1]), 
			(rect[2], rect[3]), (0, 255, 0), 2)
		cv2.imshow("Tracking", frame)

	# Break if user press "q" (Force quit with ctl+c)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()
