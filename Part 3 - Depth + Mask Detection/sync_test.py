import time
import cv2
import numpy as np

count = 0
while True:

	frame = np.zeros((256,400))
	cv2.putText(frame, str(count), (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255,255))
	cv2.imshow("test",frame)

	count += 1

	key = cv2.waitKey(1)
	if key == ord('q'):
		break
