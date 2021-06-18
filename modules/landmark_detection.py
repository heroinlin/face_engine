# -*- coding: utf-8 -*-

import abc
import os
import sys

import cv2
import numpy as np

working_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.dirname(working_root))
from modules.plot_utils import draw_landmarks
from thirdparty.face_landmark_detection import \
    PLFDONNX


class ILandmarkDetector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def detect(self, image: np.ndarray):
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


class LandmarkDetector(ILandmarkDetector):
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.detector = None
        self.init_interface()

    def init_interface(self):
        self.detector = PLFDONNX(
            model_path=self.model_path)

    def detect(self, image: np.ndarray):
        outputs = self.detector.detect(image)
        return outputs

    def set_config(self, key: str, value):
        self.detector.set_config(key, value)

    def destroy(self):
        pass


def main():
    detector = LandmarkDetector()
    image_path = os.path.join(working_root, r"../data/image_data/images/00000_1.jpg")
    image = cv2.imread(image_path)
    outputs = detector.detect(image)
    print(outputs)
    draw_landmarks(image, outputs[1])
    cv2.imshow("landmark detection", image)
    cv2.waitKey()
    cv2.destroyWindow("landmark detection")


if __name__ == '__main__':
    main()
