# USAGE
# In the command line type:
# python RedObjectTracking.py --video Videos/RedObjects.mov

# Import the necessary packages

import numpy as np
import argparse
import time # Useful for fast processing
import cv2

# Initialize argument parser & parse arguments

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", 
	help = "Path to the (optional) video file")
args = vars(ap.parse_args())

# Track Shades of blue (define limits)

redLower = np.array([0, 0, 80], dtype = "uint8")
redUpper = np.array([50, 30, 255], dtype = "uint8")

camera = cv2.VideoCapture(args["video"])

# Start looping over the frames

while True:
	
	(grabbed, frame) = camera.read()

	if not grabbed:
		
		break

	red = cv2.inRange(frame, redLower, redUpper)
	red = cv2.GaussianBlur(red, (3, 3), 0)

	# Find the contours in the thresholded image

	(cnts, _) = cv2.findContours(red.copy(), cv2.RETR_EXTERNAL, 
		cv2.CHAIN_APPROX_SIMPLE)

	# If contours were found

	if len(cnts) > 0:

		cnt = sorted(cnts, key = cv2.contourArea, 
			reverse = True)[0]

		rect = np.int32(cv2.cv.BoxPoints(cv2.minAreaRect(cnt)))
		cv2.drawContours(frame, [rect], -1, (0, 255, 0), 2)

	# Show the results

	cv2.imshow("Object Tracking", frame)
	cv2.imshow("Binary Object", red)
	
	# Optional sleep time

	time.sleep(0.025)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()
