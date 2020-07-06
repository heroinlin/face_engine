import os
import sys
import glob
import cv2
import numpy as np
import time

working_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.dirname(os.path.dirname(working_root)))
from modules.object_detection import (OnnxObjectDetector, draw_detection_rects)
from modules.face_recognition import (PyFeatureExtract, OnnxFeatureExtract)


class VideoRecognition(object):
    """
    视频目标检测
    detector  目标检测接口, 默认使用python端onnx接口
    """

    def __init__(self, person_library, detector=None, extractor=None):
        super(VideoRecognition).__init__()
        self.decay_time = 0
        self.auto_play_flag = False
        self.time_printing = True
        self.detector = detector
        self.extractor = extractor
        self.init()
        self.person_library = person_library
        self.person_id_library = None
        self.person_feature_library = None
        self.config = {"feature_size": 512}

    def init(self):
        if self.detector is None:
            # self.detector = PyObjectDetector()
            self.detector = OnnxObjectDetector()
            self.extractor = OnnxFeatureExtract()

    @staticmethod
    def normalize(nparray, order=2, axis=0):
        """Normalize a N-D numpy array along the specified axis."""
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)

    @staticmethod
    def crop_image(image, detection_rect, method=1):
        if not isinstance(detection_rect, np.ndarray):
            detection_rect = np.array(detection_rect)
        if method:
            width = image.shape[1]
            height = image.shape[0]
        else:
            width = 1.0
            height = 1.0
        box = [int(detection_rect[0] * width), int(detection_rect[1] * height),
               int(detection_rect[2] * width), int(detection_rect[3] * height)]
        image = image[box[1]:box[3], box[0]:box[2], :]
        return image

    def person_library_feature_extract(self):
        self.person_feature_library = np.zeros([0, self.config['feature_size']], np.float)
        self.person_id_library = sorted(os.listdir(self.person_library))
        for person_folder in self.person_id_library:
            person_id_features = np.zeros([0, self.config['feature_size']], np.float)
            for image_path in sorted(glob.glob(os.path.join(self.person_library, person_folder, "*.jpg"))):
                image = cv2.imdecode(np.fromfile(image_path, np.uint8()), 1)
                features_array = self.extractor.feature_extract(image)
                features_array = features_array.reshape(-1, self.config['feature_size'])
                person_id_features = np.vstack((person_id_features, features_array))
            # 取特征的均值
            person_id_features = np.mean(person_id_features, 0)
            # 归一化
            person_id_features = self.normalize(person_id_features, axis=0)
            self.person_feature_library = np.vstack((self.person_feature_library, person_id_features))

    def recognition(self, image):
        if self.person_feature_library is None:
            print("No person in the person library!")
            exit(-1)
        features_array = self.extractor.feature_extract(image)
        features_array = features_array.reshape(-1, self.config['feature_size'])
        features_array = self.normalize(features_array, axis=1)
        dist_mat = 1 - np.dot(features_array, self.person_feature_library.transpose())
        dist_sorted = np.sort(dist_mat, axis=1)
        dist_sorted_idx = np.argsort(dist_mat, axis=1)
        return self.person_id_library[dist_sorted_idx[0][0]], dist_sorted[0][0]

    @staticmethod
    def draw_person_id_on_image(image, person_id, rect, method=1):
        if method:
            width = image.shape[1]
            height = image.shape[0]
        else:
            width = 1.0
            height = 1.0
        cv2.putText(image, f"{person_id}",
                    (int(rect[0] * width), int(rect[1] * height+10)),
                    1, 1, (255, 0, 255))

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
            for rect in bounding_boxes:
                face = self.crop_image(frame.copy(), rect)
                face_id, dist = self.recognition(face)
                self.draw_person_id_on_image(frame, face_id, rect)

            end_time = time.time()
            fps = 1/(end_time - start_time)
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


def main():
    person_library = r"F:\tmp\person_search\librarys"
    work_root = os.path.dirname(os.path.dirname(working_root))
    video_path = os.path.join(work_root, "data/video_data/videos/1.mp4")
    onnx_file_path = os.path.join(work_root, r"checkpoints/face_reid/plr_osnet_237_2.1969-sim.onnx")
    detector = OnnxObjectDetector()
    extractor = OnnxFeatureExtract(onnx_file_path)
    person_search = VideoRecognition(person_library, detector, extractor)
    person_search.person_library_feature_extract()
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video: ", video_path)
        return
    person_search.detect(video)


if __name__ == '__main__':
    main()
