# -*- coding: utf-8 -*-

import abc
import os
import sys

import cv2
import numpy as np

working_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.dirname(working_root))
from thirdparty.face_detect_inference import \
    FaceDetector_mtcnn, FaceDetector_ssd, FaceDetector_onnx


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


class OnnxObjectDetector(IObjectDetector):
    def __init__(self, checkpoint_file_path=None):
        self.checkpoint_file_path = checkpoint_file_path
        self.init_interface()

    def init_interface(self):
        self.detector = FaceDetector_onnx(
            onnx_file_path=self.checkpoint_file_path)

    def detect(self, image: np.ndarray) -> list:
        bounding_boxes = self.detector.detect(image)
        return bounding_boxes

    def set_config(self, key, value):
        return super().set_config(key, value) 

    def destroy(self):
        pass


def draw_detection_rects(image: np.ndarray, detection_rects: np.ndarray, color=(0, 255, 0), method=1):
    if not isinstance(detection_rects, np.ndarray):
        detection_rects = np.array(detection_rects)
    if method:
        width = image.shape[1]
        height = image.shape[0]
    else:
        width = 1.0
        height = 1.0
    for index in range(detection_rects.shape[0]):
        cv2.rectangle(image,
                      (int(detection_rects[index, 0] * width), int(detection_rects[index, 1] * height)),
                      (int(detection_rects[index, 2] * width), int(detection_rects[index, 3] * height)),
                      color,
                      thickness=2)
        if detection_rects.shape[1] == 5:
            cv2.putText(image, f"{detection_rects[index, 4]:.03f}",
                        (int(detection_rects[index, 0] * width), int(detection_rects[index, 1] * height)),
                        1, 1, (255, 0, 255))


def main():
    detector = PyObjectDetector()
    detector.set_config("detect_threshold", 0.8)
    image_path = os.path.join(os.path.dirname(working_root), r"./data/image_data/images/00000_1.jpg")
    image = cv2.imread(image_path)
    rects = detector.detect(image)
    print(rects)
    draw_detection_rects(image, rects)
    cv2.imshow("object detection", image)
    cv2.waitKey()
    cv2.destroyWindow("object detection")


if __name__ == '__main__':
    main()
