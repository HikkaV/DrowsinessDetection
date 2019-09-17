from keras import backend as K
from PIL import Image
import cv2
import numpy as np
from utils.settings import cfg_darnket, weights_darknet
from matplotlib import pyplot as plt


def plot_pred(img_, output, ifsave=True, name='slide.png'):
    fig, ax = plt.subplots(figsize=(18, 20))
    imgplot = ax.imshow(img_)
    x = []
    y = []
    for z, i in enumerate(output):
        if z % 2 == 0:
            x.append(i)
        else:
            y.append(i)
    output = list(zip(x, y))
    for i in output:
        ax.scatter(int(i[0]), int(i[1]), 50)

    plt.title('Prediction for epoch{0}'.format(name.split('epoch')[1].replace('.png', '')))
    plt.show()
    if ifsave:
        fig.savefig(name)


class ImagePreprocessor:
    def __init__(self, cfg=cfg_darnket, weights=weights_darknet):
        self.cfg = cfg
        self.weights = weights
        self.net = cv2.dnn.readNetFromDarknet(self.cfg, self.weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def resize(self, image, landmarks, width, height):
        cur_height = image.height
        cur_width = image.width
        image = image.resize((height, width))
        for i in range(len(landmarks)):
            landmarks[i] = [
                width / cur_width * landmarks[i][0],
                height / cur_height * landmarks[i][1]
            ]
        return image, landmarks

    def crop_face(self, image, landmarks):
        param = 60
        blob = cv2.dnn.blobFromImage(image,
                                     1 / 255, (416, 416), [0, 0, 0],
                                     1,
                                     crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(ImagePreprocessor.get_outputs_names(self.net))
        faces, _, __ = ImagePreprocessor.post_process(image, outs, 0.7,
                                                      0.8)
        crop_faces = [(faces[i][0] - param, faces[i][1] - param,
                       faces[i][2] + faces[i][0] + param,
                       faces[i][3] + faces[i][1] + param)
                      for i in range(len(faces))]

        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        crop_faces = [(faces[i][0] - param, faces[i][1] - param,
                       faces[i][2] + faces[i][0] + param,
                       faces[i][3] + faces[i][1] + param)
                      for i in range(len(faces))]
        img = Image.fromarray(np.asarray(img)).crop(
            crop_faces[0]).convert('RGB')
        landmarks = [(i[0] - crop_faces[0][0], i[1] - crop_faces[0][1])
                     for i in landmarks]

        return img, landmarks

    @staticmethod
    def post_process(image, outs, conf_threshold, nms_threshold):
        image_height = image.shape[0]
        image_width = image.shape[1]

        # Scan through all the bounding boxes output from the network and keep only
        # the ones with high confidence scores. Assign the box's class label as the
        # class with the highest score.
        confidences = []
        boxes = []

        people_boxes = []
        center = []
        class_ids = []
        if True:

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > conf_threshold:
                        center_x = int(detection[0] * image_width)
                        center_y = int(detection[1] * image_height)
                        width = int(detection[2] * image_width)
                        height = int(detection[3] * image_height)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
                        center.append([center_x, center_y])
                        class_ids.append(class_id)

        # Perform non maximum suppression to eliminate redundant
        # overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                   nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]

            people_boxes.append(box)

        return people_boxes, class_ids, indices

    @staticmethod
    def get_outputs_names(net):
        layers_names = net.getLayerNames()
        return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
