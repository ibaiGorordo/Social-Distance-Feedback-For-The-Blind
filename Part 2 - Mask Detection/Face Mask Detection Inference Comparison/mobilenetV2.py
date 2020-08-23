import numpy as np
import tensorflow as tf
import cv2
import pathlib
import time
import six

from object_detection.utils import label_map_util

# Patch the location of gfile
tf.gfile = tf.io.gfile

class mobilenetV2_SSD():

	def __init__(self, model_path, label_path):
		self.load_model(model_path)
		self.load_labels(label_path)
		self.colors = [(22,22,250),(214,202,18)]

	def load_model(self, model_path):
		# Load a (frozen) Tensorflow model into memory.
		detection_graph = tf.Graph()
		with detection_graph.as_default():
			od_graph_def = tf.compat.v1.GraphDef()
			with tf.io.gfile.GFile(model_path, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
				with detection_graph.as_default():
					self.sess = tf.Session(graph=detection_graph)
					self.detection_graph = detection_graph

	def load_labels(self,label_path):
		self.category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)

	def run_inference_for_single_image(self, image_np):
		
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis=0)
		# Extract image tensor
		image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
		output_dict = {}
		# Extract detection boxes
		output_dict['detection_boxes'] = self.detection_graph.get_tensor_by_name('detection_boxes:0')
		# Extract detection scores
		output_dict['detection_scores'] = self.detection_graph.get_tensor_by_name('detection_scores:0')
		# Extract detection classes
		output_dict['detection_classes'] = self.detection_graph.get_tensor_by_name('detection_classes:0')
		# Extract number of detectionsd
		output_dict['num_detections'] = self.detection_graph.get_tensor_by_name(
			'num_detections:0')
		
		# Actual detection.
		(output_dict['detection_boxes'], output_dict['detection_scores'], output_dict['detection_classes'], output_dict['num_detections']) = self.sess.run(
			[output_dict['detection_boxes'], output_dict['detection_scores'], output_dict['detection_classes'], output_dict['num_detections']],
			feed_dict={image_tensor: image_np_expanded})

		return output_dict

	def draw_detection(self, image_np, output_dict, min_score):
		boxes = np.squeeze(output_dict['detection_boxes'])
		scores = np.squeeze(output_dict['detection_scores'])
		classes = np.squeeze(output_dict['detection_classes']).astype(np.int32)

		num_detections = boxes.shape[0]

		for (score, label, box) in zip(scores, classes, boxes):
			if scores is None or score > min_score:
				box = tuple(box.tolist())
				if label in six.viewkeys(self.category_index):
					class_name = self.category_index[label]['name']
				else:
					class_name = 'N/A'
				display_str = str(class_name)
				display_str = '{}: {}%'.format(display_str, round(100*score))

				ymin, xmin, ymax, xmax = box
				xmin = int(np.round(xmin*image_np.shape[1]))
				xmax = int(np.round(xmax*image_np.shape[1]))
				ymin = int(np.round(ymin*image_np.shape[0]))
				ymax = int(np.round(ymax*image_np.shape[0]))

				image_np = cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), self.colors[label-1], 2)
				cv2.putText(image_np, display_str, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[label-1], 2)
		return image_np

	def inference(self,image_np, min_score=0.5):
		input_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
		output_dict = self.run_inference_for_single_image(input_image)

		return self.draw_detection(image_np, output_dict, min_score)
