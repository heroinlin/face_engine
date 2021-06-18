# -*- coding: utf-8 -*-
import argparse
import glob
import os
import sys
import time
import cv2
working_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.dirname(os.path.dirname(working_root)))
from modules.object_detection import (OnnxObjectDetector, PyObjectDetector, draw_detection_rects)


class VideoObjectDetection(object):
    """
    视频目标检测
    detector  目标检测接口, 默认使用python端onnx接口
    """

    def __init__(self, detector=None):
        super(VideoObjectDetection).__init__()
        self.decay_time = 0
        self.auto_play_flag = False
        self.time_printing = True
        self.detector = detector
        self.init()

    def init(self):
        if self.detector is None:
            # self.detector = PyObjectDetector()
            self.detector = OnnxObjectDetector()

    def detect(self, video):
        """
        对单个视频的目标检测
        Parameters
        ----------
        video      cv2的VideoCapture()类

        Returns
        -------
            无
        """
        frame_num = 0
        while True:
            _, frame = video.read()
            if frame is None:
                break
            frame_num += 1
            start_time = time.time()
            if frame_num % 1 == 0:
                # bounding_boxes = [[<score>,<box>],[<score>,<box>],...],  维度为n * 5
                bounding_boxes = self.detector.detect(frame)
                # print(bounding_boxes)
            else:
                bounding_boxes = []
            draw_detection_rects(frame, bounding_boxes)
            end_time = time.time()
            if self.time_printing:
                print("predict time is {:f}".format(end_time - start_time))
                self.time_printing = False
            if "win" in sys.platform:
                cv2.imshow("object_detection", frame)
                key = cv2.waitKey(self.decay_time)
                if key == 32:
                    self.auto_play_flag = not self.auto_play_flag
                    self.decay_time = 1 if self.auto_play_flag else 0
                if key == 27:
                    break

    def destroy(self):
        self.detector.destroy()


def object_detection_for_videos_folder_test(video_folder_path: str,
                                            checkpoint_file_path: str = None,
                                            mean: list = None,
                                            stddev: list = None):
    """
        视频文件夹目标检测
        Parameters
        ----------
        test_folder_path 视频文件夹地址
        checkpoint_file_path 目标检测模型地址
        mean 目标检测模型训练使用的图像均值
        stddev 目标检测模型训练使用的图像标准差
        Returns
        -------
            None
    """
    detector = None
    if checkpoint_file_path is not None:
        detector = OnnxObjectDetector(checkpoint_file_path)
    video = cv2.VideoCapture()
    video_object_detection = VideoObjectDetection(detector=detector)
    for video_index, video_path in enumerate(
            sorted(glob.glob("{}/*.mp4".format(video_folder_path)))):
        print(video_index, os.path.basename(video_path))
        if not video.open(video_path):
            print("can not open the video: ", video_path)
            return
        video_object_detection.detect(video)
    video.release()


def object_detection_for_camera(checkpoint_file_path: str = None,
                                mean: list = None,
                                stddev: list = None):
    """
        视频文件夹目标检测
        Parameters
        ----------
        checkpoint_file_path 目标检测模型地址
        mean 目标检测模型训练使用的图像均值
        stddev 目标检测模型训练使用的图像标准差
        Returns
        -------
            None
    """
    detector = None
    if checkpoint_file_path is not None:
        detector = OnnxObjectDetector(checkpoint_file_path)
    video = cv2.VideoCapture()
    video_object_detection = VideoObjectDetection(detector=detector)
    if not video.open(0):
        print("can not open the video: ", video_path)
        return
    video_object_detection.detect(video)
    video.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--video_folder_path',
                        type=str,
                        help='video folder path',
                        default=os.path.join(os.path.dirname(os.path.dirname(working_root)),
                                             "data/video_data/videos"))
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        help='object detection checkpoint file path',
                        default=None)
    parser.add_argument('--mean',
                        type=float,
                        nargs='+',
                        help='train image mean value',
                        default=None)
    parser.add_argument('--stddev',
                        type=float,
                        nargs='+',
                        help='train image stddev value',
                        default=None)
    args = parser.parse_args()
    return args


def main():
    if "darwin" in sys.platform:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决OpenMP报错问题
    args = parse_args()
    print(args)
    if args.mean is not None:
        assert type(args.mean) == list and len(args.mean) == 3
    if args.stddev is not None:
        assert type(args.stddev) == list and len(args.stddev) == 3
    object_detection_for_camera(args.model, args.mean, args.stddev)
    # object_detection_for_videos_folder_test(args.video_folder_path, args.model,
    #                                         args.mean, args.stddev)


if __name__ == '__main__':
    main()
