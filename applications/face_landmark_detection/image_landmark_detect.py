# -*- coding: utf-8 -*-
import argparse
import glob
import os
import sys
import time
import cv2
import numpy as np
work_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.dirname(os.path.dirname(work_root)))
from modules.object_detection import OnnxObjectDetector as ObjectDetector
from modules.landmark_detection import LandmarkDetector
from modules.plot_utils import draw_detection_rects, draw_landmarks


class BoxLandmarkDetector(object):
    def __init__(self, face_detector=None, landmark_detector=None):
        self.face_detector = face_detector
        self.landmark_detector = landmark_detector
        if self.face_detector is None:
            self.face_detector = ObjectDetector()
        if self.landmark_detector is None:
            self.landmark_detector = LandmarkDetector()

    def enlarged_box(self, box):
        x1, y1, x2, y2 = box[0:4]
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        # 框扩大1.5倍
        w = min(max(1.0, 2 - 2 * w), 1.5)
        h = min(max(1.0, 2 - 2 * h), 1.5)
        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, 1), min(y2, 1)
        new_box = [x1, y1, x2, y2]
        return new_box

    @staticmethod
    def random_enlarge_box(box, width, height, scale_w=1.0, scale_h=1.0):
        """随机放大框, 当前框一定被新产生的框包含, box范围[0-width or 0-height]"""
        x1, y1, x2, y2 = box
        # x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        # # 框扩大1.5倍
        # scale_w = min(max(1.0, 2 - 2 * w), 1.5)
        # scale_h = min(max(1.0, 2 - 2 * h), 1.5)
        dx = random.uniform(0.0, 1.0)
        dy = random.uniform(0.3, 1.0)
        max_dw = (scale_w - 1.0) * (x2 - x1)
        max_dh = (scale_h - 1.0) * (y2 - y1)
        x1 = max(0, int(x1 - dx * max_dw))
        y1 = max(0, int(y1 - dy * max_dh))
        dw = random.uniform(0, 1.0 - dx)
        dh = random.uniform(0, 1.0 - dy)
        x2 = min(int(x2 + dw * max_dw + 1), width - 1)
        y2 = min(int(y2 + dh * max_dh + 1), height - 1)
        box = [x1, y1, x2, y2]
        return box

    def cut_box_with_image(self, image, box):
        x1 = int(box[0] * image.shape[1])
        y1 = int(box[1] * image.shape[0])
        x2 = int(box[2] * image.shape[1])
        y2 = int(box[3] * image.shape[0])
        return image[y1:y2, x1:x2, :]

    def box_landmark_process(self, box, landmark):
        """
        将关于box的关键点转化为关于全图的关键点
        """
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        for idx in range(landmark.shape[0]):
            landmark[idx, 0] = landmark[idx, 0] * box_width + box[0]
            landmark[idx, 1] = landmark[idx, 1] * box_height + box[1]
        return landmark

    def detect(self, image, detect_face=True):
        landmarks = []
        if detect_face:
            bounding_boxes = self.face_detector.detect(image)
            for bounding_box in bounding_boxes:
                roi_box = self.enlarged_box(bounding_box)
                roi_img = self.cut_box_with_image(image, roi_box)
                outputs = self.landmark_detector.detect(roi_img)
                landmark = self.box_landmark_process(roi_box, outputs[1][0])
                landmarks.append(landmark)
            return bounding_boxes, landmarks
        else:
            angles, landmarks = self.landmark_detector.detect(roi_img)
            return None, landmarks

    def destroy(self):
        pass


def landmark_detection_for_image():
    detector = BoxLandmarkDetector()
    image_path = os.path.join(work_root, "../../data/image_data/images/00000_1.jpg")
    image = cv2.imread(image_path, 1)
    boxes, landmarks = detector.detect(image)
    draw_detection_rects(image, boxes)
    draw_landmarks(image, landmarks)
    cv2.imshow("image", image)
    cv2.waitKey()


if __name__ == '__main__':
    landmark_detection_for_image()
