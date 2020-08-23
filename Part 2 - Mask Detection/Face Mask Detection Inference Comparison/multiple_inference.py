import cv2
import numpy as np

from mobilenetV2 import mobilenetV2_SSD
from yolov3_tiny import yolov3_tiny

mobilenetv2_label_path = 'mobilenetv2 SSD - depthai/label_map.pbtxt'
mobilenetv2_model_path = 'mobilenetv2 SSD - depthai/frozen_inference_graph.pb'

yolov3_tiny_label_path = "yolov3-tiny model - depthai/obj.names"
yolov3_tiny_config_path = "yolov3-tiny model - depthai/yolov3-tiny_obj.cfg"
yolov3_tiny_weight_path = "yolov3-tiny model - depthai/yolov3-tiny_obj_best.weights"


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("videos/production ID_4236787.mp4")

mobilenetV2_model = mobilenetV2_SSD(mobilenetv2_model_path, mobilenetv2_label_path)
yolov3_tiny_model = yolov3_tiny(yolov3_tiny_weight_path, yolov3_tiny_config_path, yolov3_tiny_label_path)

while cap.isOpened():
	ret, image_np = cap.read()
	image_np = cv2.resize(image_np,((800,600)))

	mobilenetv2_img = mobilenetV2_model.inference(image_np.copy())
	yolov3_tiny_img = yolov3_tiny_model.inference(image_np.copy())

	img_plot = cv2.resize(cv2.hconcat([mobilenetv2_img, yolov3_tiny_img]), (1000, 400))
	black_space = img_plot[:30,:,:]*0
	img_plot = cv2.vconcat([black_space,img_plot])

	cv2.putText(img_plot, "mobilenetV2-SSD", (150,22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
	cv2.putText(img_plot, "YoloV3-tiny", (700,22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
	cv2.imshow('Comparison', img_plot)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cap.release()
		cv2.destroyAllWindows()
		break



