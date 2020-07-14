import os
import sys
import glob
import cv2
import numpy as np
import time
import random

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
        self.config = {"feature_size": 512,
                       'pic_nums': 10}

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
    def random_enlarge_box(box, width, height, scale_w=1.0, scale_h=1.0):
        """随机放大框, 当前框一定被新产生的框包含, box范围[0-width or 0-height]"""
        x1, y1, x2, y2 = box
        # x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        # # 框扩大1.5倍
        # scale_w = min(max(1.0, 2 - 2 * w), 1.5)
        # scale_h = min(max(1.0, 2 - 2 * y), 1.5)
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

    def crop_image(self, image, detection_rect, method=1):
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
        box = self.random_enlarge_box(box, image.shape[1], image.shape[0], scale_w=1.1, scale_h=1.3)
        image = image[box[1]:box[3], box[0]:box[2], :]
        return image

    def person_library_feature_extract(self):
        self.person_feature_library = np.zeros([0, self.config['feature_size']], np.float)
        self.person_id_library = sorted(os.listdir(self.person_library))
        print(self.person_id_library)
        for person_folder in self.person_id_library:
            print("extract feature ", person_folder)
            person_id_features = np.zeros([0, self.config['feature_size']], np.float)
            images_list = sorted(glob.glob(os.path.join(self.person_library, person_folder, "*.[jp][pn]g")))
            if self.config['pic_nums'] >= 1:
                indeces = [int(i * max(1.0, ((len(images_list) - 1) / self.config['pic_nums'])))
                           for i in range(0, min(self.config['pic_nums'], len(images_list)))]
            else:
                indeces = range(len(images_list))
            images_list = [images_list[i] for i in indeces]
            for image_path in images_list:
                image = cv2.imdecode(np.fromfile(image_path, np.uint8()), 1)
                features_array = self.extractor.feature_extract(image)
                features_array = features_array.reshape(-1, self.config['feature_size'])
                person_id_features = np.vstack((person_id_features, features_array))
            # 取特征的均值
            if person_id_features.shape[0] > 1:
                person_id_features = np.mean(person_id_features, 0, keepdims=True)
            # 归一化
            person_id_features = self.normalize(person_id_features, axis=1)
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
        if dist_sorted[0][0] > 0.3:
            return 'unknown', 2.0
        return self.person_id_library[dist_sorted_idx[0][0]], dist_sorted[0][0]

    @staticmethod
    def draw_person_id_on_image(image, person_id, rect, method=1):
        if method:
            width = image.shape[1]
            height = image.shape[0]
        else:
            width = 1.0
            height = 1.0
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        font = ImageFont.truetype('simkai.ttf', 15)
        fillColor = (255, 0, 255)
        position = (int(rect[0] * width), int(rect[1] * height+10))
        draw = ImageDraw.Draw(pil_img)
        draw.text(position, person_id, font=font, fill=fillColor)
        image = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        return image
        # cv2.putText(image, f"{person_id}",
        #             (int(rect[0] * width), int(rect[1] * height+10)),
        #             1, 1, (255, 0, 255))

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
            for rect in bounding_boxes:
                face = self.crop_image(frame.copy(), rect, method=1)
                cv2.imshow("face", face)
                face_id, dist = self.recognition(face)
                frame = self.draw_person_id_on_image(frame, face_id, rect)
            end_time = time.time()
            fps = 1/(end_time - start_time)
            draw_detection_rects(frame, bounding_boxes, method=1)
            if "win" in sys.platform:
                cv2.imshow("object_detection", frame)
                key = cv2.waitKey(self.decay_time)
                if key == 32:
                    self.auto_play_flag = not self.auto_play_flag
                    self.decay_time = 1 if self.auto_play_flag else 0
                if key == 27:
                    break
                if key == ord('l'):
                    frame_num += 500
                    frame_num = min(frame_num, video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    def destroy(self):
        self.detector.destroy()


def main():
    person_library = r"F:\tmp\person_search\librarys"
    work_root = os.path.dirname(os.path.dirname(working_root))
    video_path = os.path.join(work_root, "data/video_data/videos/1.mp4")
    # onnx_file_path = os.path.join(work_root, r"checkpoints/face_reid/backbone_ir50_asia-sim.onnx")
    onnx_file_path = os.path.join(work_root, r"checkpoints/face_reid/plr_osnet_320_2.1876-sim.onnx")
    detector = OnnxObjectDetector()
    extractor = OnnxFeatureExtract(onnx_file_path)
    person_search = VideoRecognition(person_library, detector, extractor)
    person_search.person_library_feature_extract()
    print(person_search.person_feature_library.shape)
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video: ", video_path)
        return
    person_search.detect(video)


if __name__ == '__main__':
    main()
