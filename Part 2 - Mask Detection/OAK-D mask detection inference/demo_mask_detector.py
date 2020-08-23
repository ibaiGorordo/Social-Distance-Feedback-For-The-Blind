import numpy as np
import cv2
from demo_helpers import config, capture_image, get_detection, calculate_frame_speed, decode_mobilenet_ssd, show_mobilenet_ssd
from time import time, sleep, monotonic
import os
import depthai
print('Using depthai module from: ', depthai.__file__)

# Create a list of enabled streams ()
stream_names = ['metaout', 'previewout']

device = depthai.Device('', False)

# create the pipeline, here is the first connection with the device
p = device.create_pipeline(config=config)

if p is None:
    print('Pipeline is not created.')
    exit(3)

while True:

    # retreive data from the device
    # data is stored in packets, there are nnet (Neural NETwork) packets which have additional functions for NNet result interpretation
    nnet_packets, data_packets = p.get_available_nnet_and_data_packets(True)

    ret, frame = capture_image(data_packets)
    nnet_prev = get_detection(nnet_packets)

    if ret:
        frame_count = calculate_frame_speed()
        nn_frame = show_mobilenet_ssd(nnet_prev["entries_prev"]['rgb'], frame, is_depth=0)
        cv2.putText(nn_frame, "fps: " + str(frame_count), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
        cv2.imshow("Mask detection", nn_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.
device.deinit_device()
print('py: DONE.')

