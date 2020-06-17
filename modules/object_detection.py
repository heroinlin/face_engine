# -*- coding: utf-8 -*-

import abc
import os
import sys

import cv2
import numpy as np

from thirdparty.face_detect_inference import \
    FaceDetector_mtcnn, FaceDetector_ssd


class IObjectDetector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> list:
        """
        检测

        Parameters
        ----------
        image: np.ndarray
            图像
        Returns
        -------
            [[分数, x1, y1, x2, y2], ...]
        """
        pass

    @abc.abstractmethod
    def set_config(self, key: str, value):
        """
        配置参数

        Parameters
        ----------
        key
            键
        value
            值

        """
        pass

    @abc.abstractmethod
    def destroy(self):
        """
        释放资源
        """
        pass


class PyObjectDetector(IObjectDetector):
    def __init__(self, checkpoint_file_path=None, device=None):
        self.config = {"detect_threshold": 0.7, "nms_threshold": 0.3}
        self.checkpoint_file_path = checkpoint_file_path
        self.device = device
        self.detector = None
        self.init_interface()

    def init_interface(self):
        self.detector = FaceDetector_ssd(
            checkpoint_file_path=self.checkpoint_file_path,
            detect_threshold=self.config["detect_threshold"],
            nms_threshold=self.config["nms_threshold"],
            device=self.device)

    def detect(self, image: np.ndarray) -> list:
        bounding_boxes = self.detector.detect(image)
        return bounding_boxes

    def set_config(self, key: str, value):
        self.detector.set_config(key, value)

    def destroy(self):
        pass


def draw_detection_rects(image: np.ndarray,
                         detection_rects: list,
                         color=(0, 255, 0)):
    for rect in detection_rects:
        cv2.rectangle(
            image,
            (int(rect[1] * image.shape[1]), int(rect[2] * image.shape[0])),
            (int(rect[3] * image.shape[1]), int(rect[4] * image.shape[0])),
            color,
            thickness=2)
        cv2.putText(
            image, f"{rect[0]:.03f}",
            (int(rect[1] * image.shape[1]), int(rect[2] * image.shape[0])), 1,
            1, (255, 0, 255))


def main():
    working_root = os.path.split(os.path.realpath(__file__))[0]
    os.chdir(os.path.dirname(working_root))
    sys.path.append(os.path.dirname(working_root))

    detector = PyObjectDetector()
    detector.set_config("detect_threshold", 0.8)
    image = imread(r"./data/image_data/images/test.jpg")
    rects = detector.detect(image)
    print(rects)
    draw_detection_rects(image, rects)
    cv2.imshow("object detection", image)
    cv2.waitKey()
    cv2.destroyWindow("object detection")


if __name__ == '__main__':
    main()
