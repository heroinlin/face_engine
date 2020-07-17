# -*- coding: utf-8 -*-

import abc
import os
import cv2
import numpy as np

from thirdparty.pedestrian_detect_inference import PedestrainDetector_rfb
from thirdparty.pedestrian_detect_inference import PedestrainDetector_faster_rcnn
from thirdparty.pedestrian_detect_inference import PedestrianDetector_onnx


class IObjectDetector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        检测

        Parameters
        ----------
        image: np.ndarray
            图像
        Returns
        -------
            [[x1, y1, x2, y2, score], ...]
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


class RfbObjectDetector(IObjectDetector):
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.init_interface()

    def init_interface(self):
        self.detector = PedestrainDetector_rfb(model_path=self.model_path)

    def detect(self, image: np.ndarray) -> np.ndarray
        bounding_boxes = self.detector.detect(image)
        return np.array(bounding_boxes)

    def set_config(self, key, value):
        return self.detector.set_config(key, value)

    def destroy(self):
        pass

class OnnxObjectDetector(IObjectDetector):
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.init_interface()

    def init_interface(self):
        self.detector = PedestrianDetector_onnx(model_path=self.model_path)

    def detect(self, image: np.ndarray) -> np.ndarray
        bounding_boxes = self.detector.detect(image)
        return bounding_boxes

    def set_config(self, key, value):
        return self.detector.set_config(key, value)

    def destroy(self):
        pass

class TorchObjectDetector(IObjectDetector):
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.init_interface()

    def init_interface(self):
        self.detector = PedestrainDetector_faster_rcnn()

    def detect(self, image: np.ndarray) -> np.ndarray:
        bounding_boxes = self.detector.detect(image)
        return np.array(bounding_boxes)

    def set_config(self, key, value):
        return self.detector.set_config(key, value)

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
    detector = OnnxObjectDetector()
    detector.set_config("detect_threshold", 0.8)
    detector.set_config("nms_threshold", 1.0)
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
