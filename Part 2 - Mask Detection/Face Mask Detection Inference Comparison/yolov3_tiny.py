import cv2
import numpy as np

class yolov3_tiny():

    def __init__(self, weight_path, config_path, class_path):
        self.min_conf = 0.5
        self.min_nms = 0.4
        self.width = 416
        self.height = 416
        self.colors = [(214,202,18),(22,22,250)]
        self.classes = self.load_classes(class_path)
        self.net = self.load_model(config_path, weight_path)

    @staticmethod
    def load_classes(class_path):
        classes = None
        with open(class_path, 'rt') as file:
            classes = file.read().rstrip('\n').split('\n')

        return classes

    @staticmethod
    def load_model(config_path, weight_path):
       return cv2.dnn.readNetFromDarknet(config_path, weight_path)

    def getOutput(self, net):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds = [] 
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.min_conf:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        return (boxes, confidences, classIds)

    def draw_detection(self, frame, outs):
        boxes, confidences, classIds = self.postprocess(frame, outs)

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.min_conf, self.min_nms)
        for (box, conf, label) in zip(boxes, confidences, classIds):
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            display_str = str(self.classes[label])
            display_str = '{}: {}%'.format(display_str, round(100*conf))

            frame = cv2.rectangle(frame, (left, top), (left + width, top + height), self.colors[label], 2)
            cv2.putText(frame, display_str, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[label], 2)

        return frame

    def inference(self, image, min_score=0.5, min_nms = 0.4):
        self.min_score = 0.5
        self.min_nms = 0.4

        blob = cv2.dnn.blobFromImage(image, 1/255, (self.width, self.height), [0,0,0], 1, crop=False)
        # Sets the input to the network
        self.net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.getOutput(self.net))
        
        return self.draw_detection(image, outs)
