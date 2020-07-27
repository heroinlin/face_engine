# -*- coding: utf-8 -*-

import abc
import os
import sys

import cv2
import numpy as np

working_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.dirname(working_root))
from thirdparty.face_landmark_detect_interface import \
    PFLDLandmarkDetect


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
            [x1,y1,x2,y2,....x106,x106]
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


class PFLDLandmarkDetector(ILandmarkDetector):
    def __init__(self, model_path=None, device=None):
        self.model_path = model_path
        self.device = device
        self.detector = None
        self.init_interface()

    def init_interface(self):
        self.detector = PFLDLandmarkDetect(
            model_path=self.model_path,
            device=self.device)

    def detect(self, image: np.ndarray) -> list:
        landmarks = self.detector.detect(image)
        return landmarks

    def set_config(self, key: str, value):
        self.detector.set_config(key, value)

    def destroy(self):
        pass


def draw_landmarks(image, landmarks, norm=True):
    """

    Parameters
    ----------
    image 展示的原始图片
    landmarks 维度为[106, 2]的列表或者numpy数组
    norm 关键点坐标的归一化标记，为True表示landmark值范围为[0, 1]

    Returns
    -------

    """
    if norm:
        scale_width = image.shape[1]
        scale_height = image.shape[0]
    else:
        scale_width = 1.0
        scale_height = 1.0
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    for index in range(landmarks.shape[0]):
        pt1 = (int(scale_width * landmarks[index, 0]), int(scale_height * landmarks[index, 1]))
        cv2.circle(image, pt1, 1, (0, 0, 255), 2)
    plot_line = lambda i1, i2: cv2.line(image,
                                        (int(scale_width * landmarks[i1, 0]),
                                         int(scale_height * landmarks[i1, 1])),
                                        (int(scale_width * landmarks[i2, 0]),
                                         int(scale_height * landmarks[i2, 1])),
                                        (255, 255, 255), 1)
    close_point_list = [0, 33, 42, 51, 55, 66, 74, 76, 84, 86, 98, 106]
    for ind in range(len(close_point_list) - 1):
        l, r = close_point_list[ind], close_point_list[ind + 1]
        for index in range(l, r - 1):
            plot_line(index, index + 1)
        # 将眼部, 嘴部连线闭合
        plot_line(41, 33)  # 左眉毛
        plot_line(50, 42)  # 右眉毛
        plot_line(65, 55)  # 鼻子
        plot_line(73, 66)  # 左眼
        plot_line(83, 76)  # 右眼
        plot_line(97, 86)  # 外唇
        plot_line(105, 98)  # 内唇



def main():
    detector = PFLDLandmarkDetector()
    image_path = os.path.join(os.path.dirname(working_root), r"./data/image_data/images/00000_1.jpg")
    image = cv2.imread(image_path)
    landmarks = detector.detect(image)
    print(landmarks)
    draw_landmarks(image, landmarks)
    cv2.imshow("landmarks", image)
    cv2.waitKey()
    cv2.destroyWindow("landmarks")


if __name__ == '__main__':
    main()
