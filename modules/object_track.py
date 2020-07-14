import abc
import cv2
import numpy as np

import os
import sys
working_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.dirname(working_root))
from thirdparty.face_track_inference import \
    KCFTracker, IOUTrack


class ITracker(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def track(self, time:int, image: np.ndarray, detection_rects: list) -> list:
        """
        跟踪

        Parameters
        ----------
        image: np.ndarray
            图像
        detection_rects: list
            检测结果，[[分数, x1, y1, x2, y2], ...]
        Returns
        -------
            [{'id': id, 'detect_or_track': 0检测1跟踪, 'score': 分数, 'rect': [x1, y1, x2, y2], 'timestamp': 时间戳}, ...]
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
    def reset(self):
        """
        重置跟踪状态
        """
        pass

    @abc.abstractmethod
    def destroy(self):
        """
        释放资源
        """
        pass


class KcfTracker(ITracker):
    def __init__(self):
        self.tracker = KCFTracker()

    def init(self, image: np.ndarray, rects: list):
        return self.tracker.init(rects, image)

    def update(self, image: np.ndarray, rects: list) -> list:
        track_rects = self.tracker.update(image, rects)
        return track_rects

    def set_config(self, key, value):
        super().set_config(key, value)

    def reset(self):
        self.tracker.clear()

    def destroy(self):
        self.tracker.release()


class IouTrack(ITracker):
    def __init__(self):
        self.tracker = IOUTrack()

    def init(self, image: np.ndarray, rects: list):
        return self.tracker.track(rects, None,  image)

    def update(self, image: np.ndarray, rects: list) -> list:
        track_rects = self.tracker.track(image, None, rects)
        return track_rects

    def set_config(self, key, value):
        super().set_config(key, value)

    def reset(self):
        self.tracker.release()

    def destroy(self):
        self.tracker.release()


def draw_track_rects(image: np.ndarray, track_rects: np.ndarray, color=(0, 255, 0), method=1):
    if not isinstance(track_rects, np.ndarray):
        track_rects = np.array(track_rects)
    if method:
        width = image.shape[1]
        height = image.shape[0]
    else:
        width = 1.0
        height = 1.0
    for index in range(track_rects.shape[0]):
        cv2.rectangle(image,
                      (int(track_rects[index, 0] * width), int(track_rects[index, 1] * height)),
                      (int(track_rects[index, 2] * width), int(track_rects[index, 3] * height)),
                      color,
                      thickness=2)
        if track_rects.shape[1] == 5:
            cv2.putText(image, f"{int(track_rects[index, 4])}",
                        (int(track_rects[index, 0] * width), int((track_rects[index, 1]+track_rects[index, 3])/2 * height)),
                        1, 1, (255, 0, 255))


def main():
    tracker = CppTracker()
    image = imread(r"data/images/test.jpg")
    detection_rects = [[1.0, 0.421875, 0.5347222222222222, 0.76875, 0.9166666666666666]]
    track_rects = tracker.track(image, detection_rects)
    print(track_rects)
    draw_track_rects(image, track_rects)
    cv2.imshow("track", image)
    cv2.waitKey()
    cv2.destroyWindow("track")


if __name__ == '__main__':
    main()
