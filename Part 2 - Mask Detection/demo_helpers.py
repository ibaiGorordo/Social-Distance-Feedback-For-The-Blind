import cv2
import sys
from time import time
import json
from pathlib import Path

sys.path.insert(1, '../')
import consts.resource_paths

model_location = str(Path('models').resolve())
blob_file = str(Path(model_location, 'model.blob').absolute())
blob_file_config = str(Path(model_location, 'config.json').absolute())

with open(blob_file_config) as f:
    data = json.load(f)

labels = data['mappings']['labels']

t_start = time()
frame_count = 0
frame_count_prev = 0

nnet_prev = {}
nnet_prev["entries_prev"] = {}
nnet_prev["nnet_source"] = {}
for cam in {'rgb', 'left', 'right'}:
    nnet_prev['entries_prev'][cam] = []

# Do not modify the default values in the config Dict below directly. Instead, use the `-co` argument when running this script.
config = {
    # metaout - contains neural net output
    # previewout - color video
    'streams': ['metaout', 'previewout'],
    'depth':
    {
        'calibration_file': consts.resource_paths.calib_fpath,
        'padding_factor': 0.3,
        'depth_limit_m': 10.0, # In meters, for filtering purpose during x,y,z calc
        'confidence_threshold' : 0.5, #Depth is calculated for bounding boxes with confidence higher than this number 
    },
    'ai':
    {
        'blob_file': blob_file,
        'blob_file_config': blob_file_config,
        'calc_dist_to_bb': False,
        'keep_aspect_ratio': True,
        'camera_input': 'rgb', #'left_right'
    },
}

def show_tracklets(tracklets, frame, labels):
    # img_h = frame.shape[0]
    # img_w = frame.shape[1]

    # iterate through pre-saved entries & draw rectangle & text on image:
    tracklet_nr = tracklets.getNrTracklets()

    for i in range(tracklet_nr):
        tracklet        = tracklets.getTracklet(i)
        left_coord      = tracklet.getLeftCoord()
        top_coord       = tracklet.getTopCoord()
        right_coord     = tracklet.getRightCoord()
        bottom_coord    = tracklet.getBottomCoord()
        tracklet_id     = tracklet.getId()
        tracklet_label  = labels[tracklet.getLabel()]
        tracklet_status = tracklet.getStatus()

        # print("left: {0} top: {1} right: {2}, bottom: {3}, id: {4}, label: {5}, status: {6} "\
        #     .format(left_coord, top_coord, right_coord, bottom_coord, tracklet_id, tracklet_label, tracklet_status))
        
        pt1 = left_coord,  top_coord
        pt2 = right_coord,  bottom_coord
        color = (255, 0, 0) # bgr
        cv2.rectangle(frame, pt1, pt2, color)

        middle_pt = (int)(left_coord + (right_coord - left_coord)/2), (int)(top_coord + (bottom_coord - top_coord)/2)
        cv2.circle(frame, middle_pt, 0, color, -1)
        cv2.putText(frame, "ID {0}".format(tracklet_id), middle_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        x1, y1 = left_coord,  bottom_coord


        pt_t1 = x1, y1 - 40
        cv2.putText(frame, tracklet_label, pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        pt_t2 = x1, y1 - 20
        cv2.putText(frame, tracklet_status, pt_t2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        
    return frame

def decode_mobilenet_ssd(nnet_packet):
    detections = []
    # the result of the MobileSSD has detection rectangles (here: entries), and we can iterate through them
    for _, e in enumerate(nnet_packet.entries()):
        # for MobileSSD entries are sorted by confidence
        # {id == -1} or {confidence == 0} is the stopper (special for OpenVINO models and MobileSSD architecture)
        if e[0]['id'] == -1.0 or e[0]['confidence'] == 0.0 or e[0]['label'] > len(labels):
            break
        # save entry for further usage (as image package may arrive not the same time as nnet package)
        # the lower confidence threshold - the more we get false positives
        if e[0]['confidence'] > config['depth']['confidence_threshold']:
            # Temporary workaround: create a copy of NN data, due to issues with C++/python bindings
            copy = {}
            copy[0] = {}
            copy[0]['id']         = e[0]['id']
            copy[0]['left']       = e[0]['left']
            copy[0]['top']        = e[0]['top']
            copy[0]['right']      = e[0]['right']
            copy[0]['bottom']     = e[0]['bottom']
            copy[0]['label']      = e[0]['label']
            copy[0]['confidence'] = e[0]['confidence']
            if config['ai']['calc_dist_to_bb']:
                copy[0]['distance_x'] = e[0]['distance_x']
                copy[0]['distance_y'] = e[0]['distance_y']
                copy[0]['distance_z'] = e[0]['distance_z']
            detections.append(copy)
    return detections


def nn_to_depth_coord(x, y):
    x_depth = int(nn2depth['off_x'] + x * nn2depth['max_w'])
    y_depth = int(nn2depth['off_y'] + y * nn2depth['max_h'])
    return x_depth, y_depth

def average_depth_coord(pt1, pt2):
    factor = 1 - config['depth']['padding_factor']
    x_shift = int((pt2[0] - pt1[0]) * factor / 2)
    y_shift = int((pt2[1] - pt1[1]) * factor / 2)
    avg_pt1 = (pt1[0] + x_shift), (pt1[1] + y_shift)
    avg_pt2 = (pt2[0] - x_shift), (pt2[1] - y_shift)
    return avg_pt1, avg_pt2

def show_mobilenet_ssd(entries_prev, frame, is_depth=0):
    img_h = frame.shape[0]
    img_w = frame.shape[1]

    # iterate through pre-saved entries & draw rectangle & text on image:
    for e in entries_prev:
        if is_depth:
            pt1 = nn_to_depth_coord(e[0]['left'],  e[0]['top'])
            pt2 = nn_to_depth_coord(e[0]['right'], e[0]['bottom'])
            color = (255, 0, 0) # bgr
            avg_pt1, avg_pt2 = average_depth_coord(pt1, pt2, config)
            cv2.rectangle(frame, avg_pt1, avg_pt2, color)
            color = (255, 255, 255) # bgr
        else:
            pt1 = int(e[0]['left']  * img_w), int(e[0]['top']    * img_h)
            pt2 = int(e[0]['right'] * img_w), int(e[0]['bottom'] * img_h)
            color = (0, 0, 255) # bgr

        x1, y1 = pt1

        cv2.rectangle(frame, pt1, pt2, color)
        # Handles case where TensorEntry object label is out if range
        if e[0]['label'] > len(labels):
            print("Label index=",e[0]['label'], "is out of range. Not applying text to rectangle.")
        else:
            pt_t1 = x1, y1 + 20
            cv2.putText(frame, labels[int(e[0]['label'])], pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            pt_t2 = x1, y1 + 40
            cv2.putText(frame, '{:.2f}'.format(100*e[0]['confidence']) + ' %', pt_t2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
            if config['ai']['calc_dist_to_bb']:
                pt_t3 = x1, y1 + 60
                cv2.putText(frame, 'x:' '{:7.3f}'.format(e[0]['distance_x']) + ' m', pt_t3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                pt_t4 = x1, y1 + 80
                cv2.putText(frame, 'y:' '{:7.3f}'.format(e[0]['distance_y']) + ' m', pt_t4, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                pt_t5 = x1, y1 + 100
                cv2.putText(frame, 'z:' '{:7.3f}'.format(e[0]['distance_z']) + ' m', pt_t5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
    return frame

def capture_image(data_packets):
    ret = 0
    frame = 0
    global frame_count

    for packet in data_packets:
        packetData = packet.getData()
        if packetData is None:
            print('Invalid packet data!')
            continue
        elif packet.stream_name == 'previewout':
            
            # the format of previewout image is CHW (Chanel, Height, Width), but OpenCV needs HWC, so we
            # change shape (3, 300, 300) -> (300, 300, 3)
            data0 = packetData[0,:,:]
            data1 = packetData[1,:,:]
            data2 = packetData[2,:,:]
            frame = cv2.merge([data0, data1, data2])
            ret = 1
        frame_count += 1

    return ret, frame

def get_detection(nnet_packets):
    for _, nnet_packet in enumerate(nnet_packets):
        meta = nnet_packet.getMetadata()
        camera = 'rgb'
        if meta != None:
            camera = meta.getCameraName()
        nnet_prev["nnet_source"][camera] = nnet_packet
        nnet_prev["entries_prev"][camera] = decode_mobilenet_ssd(nnet_packet)
    return nnet_prev


def calculate_frame_speed():
    global t_start, frame_count_prev, frame_count
    t_curr = time()
    if t_start + 1.0 < t_curr:
        t_start = t_curr
            
        frame_count_prev = frame_count
        frame_count = 0
    return frame_count_prev