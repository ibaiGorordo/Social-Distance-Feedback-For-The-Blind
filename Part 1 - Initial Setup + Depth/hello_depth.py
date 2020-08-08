import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # access the camera and its data packets
import consts.resource_paths  # load paths to depthai resources
import os

device = depthai.Device("", False)

# Create the pipeline using the 'depth_sipp' stream, establishing the first connection to the device.
pipeline = device.create_pipeline(config={
    'streams': ['depth_sipp'],
    'ai': {
        "blob_file": consts.resource_paths.blob_fpath,
        "blob_file_config": consts.resource_paths.blob_config_fpath,
    }
})

if pipeline is None:
    raise RuntimeError('Pipeline creation failed!')

while True:
    # Retrieve data packets from the device.
    # A data packet contains the video frame data.
    nnet_packets, data_packets = pipeline.get_available_nnet_and_data_packets()

    for packet in data_packets:
        if packet.stream_name.startswith('depth'):
            frame = packet.getData()
            frame[frame > 60000] = 0
            frame = (frame // 30).astype(np.uint8)
            #colorize depth map, comment out code below to obtain grayscale
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            cv2.imshow(packet.stream_name, frame)

    if cv2.waitKey(1) == ord('q'):
        break

# The pipeline object should be deleted after exiting the loop. Otherwise device will continue working.
# This is required if you are going to add code after exiting the loop.
del pipeline
os._exit(0)
