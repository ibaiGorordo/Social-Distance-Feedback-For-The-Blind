import numpy as np
import cv2
from time import time, sleep, monotonic
import os
import depthai
import datetime
import pandas as pd
import consts.resource_paths

frame_info = []

def step_norm(value):
    return round(value / 0.03) * 0.03

def tst(packet):
    return packet.getMetadata().getTimestamp()

def capture_image(data_packets):
    ret = 0
    frame = 0
    global frame_info

    for packet in data_packets:
        packetData = packet.getData()

        if packetData is None:
            print('Invalid packet data!')
            continue
        
        packet_num = packet.getMetadata().getSequenceNum()
        
        frame_info.append([packet_num, packet.stream_name])
        print(packet_num, packet.stream_name)
        if packet.stream_name == 'previewout':
            
            # the format of previewout image is CHW (Chanel, Height, Width), but OpenCV needs HWC, so we
            # change shape (3, 300, 300) -> (300, 300, 3)
            data0 = packetData[0,:,:]
            data1 = packetData[1,:,:]
            data2 = packetData[2,:,:]
            frame = cv2.merge([data0, data1, data2])

            img_path = store_path + str(round(step_norm(tst(packet)),2)) + "_" + packet.stream_name + ".jpg"
            cv2.imwrite(img_path, frame)

        elif packet.stream_name == 'left' or packet.stream_name == 'right' or packet.stream_name == 'disparity':
            frame = packetData

            img_path = store_path + str(round(step_norm(tst(packet)),2))  + "_" + packet.stream_name + ".jpg"
            cv2.imwrite(img_path, frame)

model_name = "person-vehicle-bike-detection-crossroad-1016"

cnn_model_path = consts.resource_paths.nn_resource_path + model_name + "/" + model_name
blob_file = cnn_model_path + ".blob"
blob_file_config = cnn_model_path + ".json"

# Do not modify the default values in the config Dict below directly. Instead, use the `-co` argument when running this script.
config = {
    # Possible streams:
    # ['left', 'right','previewout', 'metaout', 'depth_raw', 'disparity', 'disparity_color']
    # If "left" is used, it must be in the first position.
    # To test depth use:
    # 'streams': [{'name': 'depth_raw', "max_fps": 12.0}, {'name': 'previewout', "max_fps": 12.0}, ],
    'streams': ['left', 'right', 'previewout'],
    'depth':
    {
        'calibration_file': consts.resource_paths.calib_fpath,
        'padding_factor': 0.3,
        'depth_limit_m': 10.0, # In meters, for filtering purpose during x,y,z calc
        'confidence_threshold' : 0.5, #Depth is calculated for bounding boxes with confidence higher than this number
        'median_kernel_size': 3,
        'lr_check': False,
    },
    'ai':
    {
        'blob_file': blob_file,
        'blob_file_config': blob_file_config,
        'keep_aspect_ratio': True,
        'shaves': 7,
        'cmx_slices': 7,
        'NN_engines': 1,
    },
    'camera':
    {
        'rgb':
        {
            # 3840x2160, 1920x1080
            # only UHD/1080p/30 fps supported for now
            'resolution_h': 1080,
            'fps': 30.0,
        },
        'mono':
        {
            # 1280x720, 1280x800, 640x400 (binning enabled)
            'resolution_h': 400,
            'fps': 30.0,
        },
    },
    'app':
    {
        'sync_video_meta_streams': False,
    },
}

if not depthai.init_device(consts.resource_paths.device_cmd_fpath, ""):
    print("Error initializing device. Try to reset it.")
    exit(1)

# create the pipeline, here is the first connection with the device
p = depthai.create_pipeline(config=config)

if p is None:
    print('Pipeline is not created.')
    exit(3)

base_path = "raw_data/"
store_path = base_path + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "/"
if not os.path.isdir(store_path):
    if not os.path.isdir(base_path):
        os.mkdir(base_path)
    os.mkdir(store_path)

while True:
    try:
        # retreive data from the device
        # data is stored in packets, there are nnet (Neural NETwork) packets which have additional functions for NNet result interpretation
        nnet_packets, data_packets = p.get_available_nnet_and_data_packets()

        capture_image(data_packets)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except:
        df = pd.DataFrame(frame_info, columns = ['Count', 'name'])
        df.to_csv(store_path+"info.csv",index=False)
        break
            
del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.
depthai.deinit_device()
print('py: DONE.')