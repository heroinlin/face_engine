import abc
import os
import sys

import numpy as np

working_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.dirname(working_root))
from thirdparty.face_quality_judge_inference import NRSS, Laplacian, judge_side_face


class IFaceJudge(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def judge(self, image: np.ndarray):
        """
        检测

        Parameters
        ----------
        image: np.ndarray
            图像
        facial_pts: np.ndarray
            五点关键点
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


class BlurJudge(IFaceJudge):
    def __init__(self):
        self.config = {
            'width': 112,
            'height': 112,
            'laplacian_thresh': 0.2,  # 越大越清晰
            'nrss_thresh': 0.15,  # 越大越清晰
            'dist_rate': 1.0,  # 越接近越好
            'high_ratio_variance': 0.2,  # 越小越好
            'width_ratio_variance': 0.2,  # 越小越好
        }

    def judge(self, image: np.ndarray):
        nrss_value = NRSS()(image)
        if nrss_value > self.config['nrss_thresh']:
            return True
        else:
            return False

    def side_judge(self, facial_landmarks):
        dist_rate, high_ratio_variance, width_ratio_variance = judge_side_face(facial_landmarks)
        if abs(dist_rate - self.config['dist_rate']) < 0.2 \
                and high_ratio_variance < self.config['high_ratio_variance'] \
                and width_ratio_variance < self.config['width_ratio_variance']:
            return True
        else:
            return False

    def set_config(self, key: str, value):
        self.config[key] = value

    def destroy(self):
        pass


