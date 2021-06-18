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


class BoxLandmarkVideoDetector(object):
    def __init__(self, face_detector=None, landmark_detector=None):
        self.auto_play_flag = True
        self.decay_time = 1 if self.auto_play_flag else 0
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

    def detect(self, video, detect_face=True):
        frame_num = 0
        while True:
            _, frame = video.read()
            if frame is None:
                break
            frame_num += 1
            start_time = time.time()
            if frame_num % 1 == 0:
                bounding_boxes = self.face_detector.detect(frame)
            else:
                bounding_boxes = []
            landmarks = []
            for bounding_box in bounding_boxes:
                roi_box = self.enlarged_box(bounding_box)
                roi_img = self.cut_box_with_image(frame, roi_box)
                outputs = self.landmark_detector.detect(roi_img)
                landmark = self.box_landmark_process(roi_box, outputs[1][0])
                landmarks.append(landmark)
            draw_detection_rects(frame, bounding_boxes)
            draw_landmarks(frame, landmarks)
            if "win" in sys.platform:
                cv2.imshow("image", frame)
                key = cv2.waitKey(self.decay_time)
                if key == 32:
                    self.auto_play_flag = not self.auto_play_flag
                    self.decay_time = 1 if self.auto_play_flag else 0
                if key == 27:
                    break

    def destroy(self):
        pass


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
    face_detector = None
    if checkpoint_file_path is not None:
        face_detector = ObjectDetector(checkpoint_file_path)
    video = cv2.VideoCapture()
    video_object_detection = BoxLandmarkVideoDetector(face_detector=face_detector)
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
    video = cv2.VideoCapture()
    video_object_detection = BoxLandmarkVideoDetector()
    if not video.open(0):
        print("can not open the camera!")
        return
    video_object_detection.detect(video)
    video.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--video_folder_path',
                        type=str,
                        help='video folder path',
                        default=os.path.join(os.path.dirname(os.path.dirname(work_root)),
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
