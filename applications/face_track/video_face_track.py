# -*- coding: utf-8 -*-
import argparse
import glob
import os
import sys
import time
import cv2
working_root = os.path.dirname(os.path.dirname(os.path.split(os.path.realpath(__file__))[0]))
sys.path.append(working_root)
from modules.object_detection import OnnxObjectDetector
from modules.object_track import (draw_track_rects)
from thirdparty.face_track_inference import Sort


def box_transform(bounding_boxes, width, height):
    """
    bounding_boxes  [[score, box],[score, box]]
    box框结果值域由[0,1],[0,1] 转化为[0,width]和[0,height]
    """
    for i in range(len(bounding_boxes)):
        x1 = float(bounding_boxes[i][0])
        y1 = float(bounding_boxes[i][1])
        x2 = float(bounding_boxes[i][2])
        y2 = float(bounding_boxes[i][3])
        bounding_boxes[i][0] = x1 * width
        bounding_boxes[i][1] = y1 * height
        bounding_boxes[i][2] = x2 * width
        bounding_boxes[i][3] = y2 * height
    return bounding_boxes


class VideoTrack():
    def __init__(self):
        self.detector = OnnxObjectDetector()
        self.tracker = Sort()
        self.auto_play_flag = False
        self.decay_time = 1 if self.auto_play_flag else 0

    def detect(self, video):
        frame_num = 0
        while True:
            _, frame = video.read()
            if frame is None:
                break
            frame_num += 1
            if frame_num % 1 == 0:
                # bounding_boxes = [[<score>,<box>],[<score>,<box>],...],  维度为n * 5
                bounding_boxes = self.detector.detect(frame)
                # print(bounding_boxes)
            else:
                bounding_boxes = []
            bounding_boxes = box_transform(bounding_boxes, frame.shape[1], frame.shape[0])
            track_boxes = self.tracker.update(bounding_boxes, frame.shape[0:2], '', range(len(bounding_boxes)), 2)
            draw_track_rects(frame, track_boxes, method=0)
            if "win" in sys.platform:
                cv2.imshow("track", frame)
                key = cv2.waitKey(self.decay_time)
                if key == 32:
                    self.auto_play_flag = not self.auto_play_flag
                    self.decay_time = 1 if self.auto_play_flag else 0
                if key == 27:
                    break


if __name__ == '__main__':
    video_tracker = VideoTrack()
    video_path = os.path.join(working_root, "data/video_data/videos/1.mp4")
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("video open failure!")
        exit(-1)
    video_tracker.detect(video)