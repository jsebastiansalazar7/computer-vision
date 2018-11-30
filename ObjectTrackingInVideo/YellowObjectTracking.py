# USAGE
# In the command line type:
# python YellowObjectTracking.py --video Videos/YellowObjects.mov

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

yellowLower = np.array([0, 128, 155], dtype = "uint8")
yellowUpper = np.array([100, 255, 255], dtype = "uint8")

camera = cv2.VideoCapture(args["video"])

# Start looping over the frames

while True:
	
	(grabbed, frame) = camera.read()

	if not grabbed:
		
		break

	yellow = cv2.inRange(frame, yellowLower, yellowUpper)
	yellow = cv2.GaussianBlur(yellow, (3, 3), 0)

	# Find the contours in the thresholded image

	(cnts, _) = cv2.findContours(yellow.copy(), cv2.RETR_EXTERNAL, 
		cv2.CHAIN_APPROX_SIMPLE)

	# If contours were found

	if len(cnts) > 0:

		cnt = sorted(cnts, key = cv2.contourArea, 
			reverse = True)[0]

		rect = np.int32(cv2.cv.BoxPoints(cv2.minAreaRect(cnt)))
		cv2.drawContours(frame, [rect], -1, (0, 255, 0), 2)

	# Show the results

	cv2.imshow("Object Tracking", frame)
	cv2.imshow("Binary Object", yellow)
	
	# Optional sleep time

	time.sleep(0.025)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()
