from keras import backend as K
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import pandas as pd


class Process:
    def __init__(self, cfg, weights, path_to_save_img=None, path_to_df=None):
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.path_to_save = path_to_save_img
        self.df = pd.read_csv(path_to_df) if path_to_df else None

    # def unlist_lands(self, landmarks):
    #     lands = []
    #     for i in landmarks:
    #         lands.append(i[0])
    #         lands.append(i[1])
    #     return lands

    def resize_image_landmarks(self, image, new_height, new_width, landmarks):
        cur_height = image.height
        cur_width = image.width
        image = image.resize((new_height, new_width))
        for i in range(len(landmarks)):
            landmarks[i] = (
                new_width / cur_width * landmarks[i][0],
                new_height / cur_height * landmarks[i][1]
            )
        return image, landmarks

    def crop_only_image(self, image):
        param = 20
        blob = cv2.dnn.blobFromImage(np.array(image), 1 / 255, (416, 416),
                                     [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_outputs_names(self.net))
        faces, _, __ = self.post_process(np.array(image), outs, 0.7, 0.8)
        faces = faces[0] if faces[0][0] > faces[0][1] else faces[1]
        crop_faces = [faces[0] - param, faces[1] - param, faces[2] + faces[0] + param, faces[3] + faces[1] + param]
        img = Image.fromarray(image).crop(crop_faces)
        return img, (crop_faces[0], crop_faces[1])

    def get_outputs_names(self, net):
        layers_names = net.getLayerNames()
        return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def post_process(self, frame, outs, conf_threshold, nms_threshold):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
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
                        center_x = int(detection[0] * frame_width)
                        center_y = int(detection[1] * frame_height)
                        width = int(detection[2] * frame_width)
                        height = int(detection[3] * frame_height)
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


def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_true-y_pred))

def plot_pred(img_, output, ifsave=True, name='slide.png'):
    fig, ax = plt.subplots(figsize=(18, 20))
    imgplot = ax.imshow(img_)
    print(len(output))
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

    plt.title('Prediction')
    plt.show()
    if ifsave:
        fig.savefig(name)
