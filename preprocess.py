import cv2
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np


class Process:
    def __init__(self, cfg, weights, path_to_save_img=None, path_to_df=None):
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.path_to_save = path_to_save_img
        self.df = pd.read_csv(path_to_df) if path_to_df else None

    def unlist_lands(self, landmarks):
        lands = []
        for i in landmarks:
            lands.append(i[0])
            lands.append(i[1])
        return lands

    def make_new_data(self):
        list_ = []
        for i in tqdm(self.df.iterrows()):
            img = Image.open(i[1][0])
            img, landmarks = self.crop(img, i[1][3:])
            cv2.imwrite(self.path_to_save + i[1][0].split('/')[-1], img)
            landmarks = self.unlist_lands(landmarks)
            list_.append((self.path_to_save + i[1][0].split('/')[-1], i[1][1], i[1][2], *landmarks))
        names = ['im_name', 'height', 'width'] + list(self.df.columns[3:])
        print(names)
        new_df = pd.DataFrame(data=list_, columns=names)

        return new_df

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
        param = 100
        blob = cv2.dnn.blobFromImage(np.array(image), 1 / 255, (416, 416),
                                     [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_outputs_names(self.net))
        faces, _, __ = self.post_process(np.array(image), outs, 0.7, 0.8)
        crop_faces = [(faces[i][0] - param, faces[i][1] - param, faces[i][2] + faces[i][0] + param,
                       faces[i][3] + faces[i][1] + param) for i in
                      range(len(faces))]
        img = Image.fromarray(image).crop(crop_faces[0])
        return img, (crop_faces[0][0], crop_faces[0][1])

    def crop(self, image, landmarks):
        param = 60
        blob = cv2.dnn.blobFromImage(np.array(image), 1 / 255, (416, 416),
                                     [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_outputs_names(self.net))
        faces, _, __ = self.post_process(np.array(image), outs, 0.7, 0.8)
        crop_faces = [(faces[i][0] - param, faces[i][1] - param, faces[i][2] + faces[i][0] + param,
                       faces[i][3] + faces[i][1] + param) for i in
                      range(len(faces))]
        landmarks = list(zip(
            landmarks[[i for i in range(0, 34, 1) if i % 2 == 0]],
            landmarks[[i for i in range(0, 34, 1) if i % 2 != 0]]))
        landmarks = [(i[0] - crop_faces[0][0], i[1] - crop_faces[0][1]) for i in landmarks]
        img = image.crop(crop_faces[0])
        img, landmarks = self.resize_image_landmarks(img, 128, 128, landmarks)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return np.array(img), landmarks

    def get_outputs_names(self, net):
        layers_names = net.getLayerNames()
        return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def post_process(self, frame, outs, conf_threshold, nms_threshold):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

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
