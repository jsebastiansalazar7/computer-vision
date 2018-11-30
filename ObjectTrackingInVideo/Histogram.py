# USAGE
# Type in the command line: python Histogram.py --video Videos/RedObjects.mov

# Importing necessary packages

from matplotlib import pyplot as plt
import numpy as np
import argparse
import time
import cv2

# Initialize Argument Parser and parse arguments

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",  
	help = "Path to the (optional) video file")
args = vars(ap.parse_args())

# Load video

video = cv2.VideoCapture(args["video"])

# Start looping over the frames

while True:

	(grabbed, frame) = video.read()

	if not grabbed:

		break

	# Split the image into its 3 channels

	chans = cv2.split(frame)
	time.sleep(0.1)
	colors = ("b", "g", "r")
	
	for (chan, color) in zip(chans, colors):
		hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
		time.sleep(0.150)
		plt.figure()
		plt.title("'Flattened' Color Histogram")
		plt.xlabel("Bins")
		plt.ylabel("# of Pixels")
		plt.plot(hist, color = color)
		plt.xlim([0, 256])
		plt.show()

	# Show the results

	cv2.imshow("Object Tracking", frame)
	
	# Optional sleep time

	time.sleep(0.050)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

video.release()
cv2.destroyAllWindows()
