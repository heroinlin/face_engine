# -*- coding: utf-8 -*-

import abc
import os
import sys

import cv2
import numpy as np

from thirdparty.face_recognition import \
    FeatureExtract, ONNXFeatureExtract


class IFeatureExtract(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def feature_extract(self, image: np.ndarray) -> list:
        """
        特征提取

        Parameters
        ----------
        image: np.ndarray
            图像
        Returns
        -------
            特征
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


class PyFeatureExtract(IFeatureExtract):
    def __init__(self, model_path=None, device=None):
        self.model_path = model_path
        self.device = device
        self.extractor = None
        self.init_interface()

    def init_interface(self):
        self.extractor = FeatureExtract(
            model_path=self.model_path, device=self.device)

    def feature_extract(self, image: np.ndarray) -> list:
        feature = self.extractor.feature_extract(image)
        return feature

    def set_config(self, key: str, value):
        self.detector.set_config(key, value)

    def destroy(self):
        pass


class OnnxFeatureExtract(IFeatureExtract):
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.extractor = None
        self.init_interface()

    def init_interface(self):
        self.extractor = ONNXFeatureExtract(
            model_path=self.model_path)

    def feature_extract(self, image: np.ndarray) -> list:
        feature = self.extractor.feature_extract(image)
        return feature

    def destroy(self):
        pass


def main():
    working_root = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(os.path.dirname(working_root))

    extractor = OnnxFeatureExtract()
    image = cv2.imread(r"./data/image_data/images/test.jpg")
    feature = extractor.feature_extract(image)
    print(feature)
   

if __name__ == '__main__':
    main()
