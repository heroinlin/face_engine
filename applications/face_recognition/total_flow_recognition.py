import cv2
import numpy as np
import random
import os
import json
import sys
import time
import glob
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

working_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.dirname(os.path.dirname(working_root)))
from modules.face_align import PtFaceAlign
from modules.object_detection import RetinaFaceDetector as FaceDetector, OnnxObjectDetector, draw_detection_rects
from modules.face_recognition import OnnxFeatureExtract as FeatureExtract
from modules.object_track import SortTrack as Tracker, draw_track_rects
from modules.face_quility_judge import BlurJudge
from modules.pedestrian_detection import OnnxObjectDetector as PedestrianDetector
# from modules.pedestrian_detection import TorchObjectDetector as PedestrianDetector


class VideoRecognition(object):
    """
    视频目标检测
    detector  目标检测接口, 默认使用python端onnx接口
    """

    def __init__(self, person_library, face_judge=None, face_align=None, tracker=None, pedestrain_detector=None,
                 face_detector=None, extractor=None):
        super(VideoRecognition).__init__()
        self.auto_play_flag = True
        self.decay_time = 1 if self.auto_play_flag else 0
        self.time_printing = True
        self.face_judge = face_judge
        self.face_align = face_align
        self.tracker = tracker
        self.pedestrain_detector = pedestrain_detector
        self.face_detector = face_detector
        self.extractor = extractor
        self.init()
        self.person_library = person_library
        self.person_id_library = list()
        self.person_feature_library = None
        self.current_persons = {}
        self.config = {
            'width': 1080,
            'height': 720,
            'feature_size': 512,
            'pic_nums': 20,
            'mean_feature': True,
            'recog_score': 0.3
        }

    def init(self):
        if self.face_judge is None:
            self.face_judge = BlurJudge()
        if self.face_align is None:
            self.face_align = PtFaceAlign()
        if self.tracker is None:
            self.tracker = Tracker()
        if self.pedestrain_detector is None:
            self.pedestrain_detector = PedestrianDetector()
        if self.face_detector is None:
            self.face_detector = FaceDetector()
        if self.extractor is None:
            self.extractor = FeatureExtract()

    def set_config(self, width, height):
        self.config['width'] = width
        self.config['height'] = height
        self.tracker.set_config('width', width)
        self.tracker.set_config('height', height)

    @staticmethod
    def normalize(nparray, order=2, axis=0):
        """Normalize a N-D numpy array along the specified axis."""
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)

    @staticmethod
    def enlarge_box(box, w_scale=1.2, h_scale=1.0):
        x1, y1, x2, y2 = box[0:4]
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        w = min(w * w_scale, 1.0)
        h = min(h * h_scale, 1.0)
        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, 1), min(y2, 1)
        box[0:4] = [x1, y1, x2, y2]
        return box

    @staticmethod
    def random_enlarge_box(box, width, height, scale_w=1.0, scale_h=1.0):
        """随机放大框, 当前框一定被新产生的框包含, box范围[0-width or 0-height]"""
        x1, y1, x2, y2 = box[0:4]
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
        box[0:4] = [x1, y1, x2, y2]
        return box

    def crop_face_image(self, image, detection_rect, method=1):
        """
        从原图像截取需要的人脸框部分
        Parameters
        ----------
        image  原图像
        detection_rect 需要截取的人脸框
        method detection_rect为相对值或绝对值

        Returns
        -------

       """
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
        # box = self.random_enlarge_box(box, image.shape[1], image.shape[0], scale_w=1.3, scale_h=1.3)
        crop_image = image[box[1]:box[3], box[0]:box[2], :]
        return crop_image

    def crop_pedestrain_box(self, detection_rect):
        """行人框获取需要截取的部分"""
        if not isinstance(detection_rect, np.ndarray):
            detection_rect = np.array(detection_rect)
        width = detection_rect[2] - detection_rect[0]
        height = detection_rect[3] - detection_rect[1]
        h_w_rate = max(1.0, height / width - 1 / 12)
        box = [detection_rect[0], detection_rect[1] - height / 12,
               detection_rect[2], detection_rect[1] + height / h_w_rate]
        box[0] = max(0, box[0])
        box[1] = max(0, box[1])
        box[2] = min(box[2], 1.0)
        box[3] = min(box[3], 1.0)
        return box

    def crop_pedestrain_image(self, image, detection_rect, method=1):
        """
        从原图像截取需要的行人框部分
        Parameters
        ----------
        image  原图像
        detection_rect 需要截取的行人框
        method detection_rect为相对值或绝对值

        Returns
        -------

        """
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
        crop_image = image[box[1]:box[3], box[0]:box[2], :]
        return crop_image

    @staticmethod
    def face_box_transform(person_box, face_boxes, landmarks=None):
        """
        将关于行人框的人脸框坐标(及关键点坐标)转化为全图的人脸框坐标
        Parameters
        ----------
        person_box
            np.ndarray 行人框
            [x1, y1, x2, y2, scores]
        face_boxes
            np.ndarray 人脸框
            [[x1, y1, x2, y2, scores], ...]
        landmark
            np.ndarray 人脸关键点
            [[left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, left_mouth_x, right_mouth_y],...]
        Returns
        -------
            关于全图的人脸框坐标(及关键点坐标)
        """
        person_width = person_box[2] - person_box[0]
        person_height = (person_box[3] - person_box[1])
        face_boxes[:, 0] = face_boxes[:, 0] * person_width + person_box[0]
        face_boxes[:, 1] = face_boxes[:, 1] * person_height + person_box[1]
        face_boxes[:, 2] = face_boxes[:, 2] * person_width + person_box[0]
        face_boxes[:, 3] = face_boxes[:, 3] * person_height + person_box[1]
        if landmarks is not None:
            landmarks[:, [0, 2, 4, 6, 8]] = person_box[0] + landmarks[:, [0, 2, 4, 6, 8]] * person_width
            landmarks[:, [1, 3, 5, 7, 9]] = person_box[1] + landmarks[:, [1, 3, 5, 7, 9]] * person_height
            return face_boxes, landmarks
        return face_boxes

    def face_lankmark_transform(self, face_box, landmark, method=1):
        """
        将关于全图的关键点坐标转化为关于人脸的坐标
        Parameters
        ----------
        face_box
            np.ndarray 人脸框
            [x1, y1, x2, y2, scores]
        landmark
            np.ndarray 人脸关键点
            [left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, left_mouth_x, right_mouth_y]
        method
            宽高使用相对值或绝对值
        Returns
        -------
            关于人脸的关键点坐标
        """
        if method:
            width = 1.0
            height = 1.0
        else:
            width = self.config['width']
            height = self.config['height']
        landmark[::2] = (landmark[::2] - face_box[0]) * width
        landmark[1::2] = (landmark[1::2] - face_box[1]) * height
        if not method:
            landmark = landmark.astype(np.int8)
        return landmark

    def face_boxes_filter(self, face_image, boxes):
        """
        人脸过滤
        Parameters
        ----------
        face_image  人脸图像
        boxes  人脸框

        Returns
        -------
            True  or False
        """
        box_width = boxes[2] - boxes[0]
        box_height = boxes[3] - boxes[1]
        if box_width < 45 / self.config['width'] or box_height < 45 / self.config['height']:
            return False
        self.face_judge.set_config('nrss_thresh', 0.10)
        if not self.face_judge.judge(cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)):
            return False
        return True

    def person_library_feature_extract(self):
        """对已有人脸库的人脸图像进行特征提取,制作人脸查询库"""
        self.person_feature_library = np.zeros([0, self.config['feature_size']], np.float)
        for person_folder in sorted(os.listdir(self.person_library)):
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
            if person_id_features.shape[0] > 1 and self.config['mean_feature']:
                person_id_features = np.mean(person_id_features, 0, keepdims=True)
            # 归一化
            person_id_features = self.normalize(person_id_features, axis=1)
            self.person_id_library.extend([person_folder] * person_id_features.shape[0])
            self.person_feature_library = np.vstack((self.person_feature_library, person_id_features))

    def recognition(self, image):
        """
        对输入人脸图像进行特征提, 同时与已有人脸查询库进行匹配查询
        Parameters
        ----------
        image 人脸图像

        Returns
        -------
            人脸id, 识别正确的概率
        """
        if self.person_feature_library is None:
            print("No person in the person library!")
            exit(-1)
        features_array = self.extractor.feature_extract(image)
        features_array = features_array.reshape(-1, self.config['feature_size'])
        features_array = self.normalize(features_array, axis=1)
        # dist_mat = 1 - np.dot(features_array, self.person_feature_library.transpose())
        # dist_sorted = np.sort(dist_mat, axis=1)
        # dist_sorted_idx = np.argsort(dist_mat, axis=1)
        # recog_score = (dist_sorted[0][0] - self.config['recog_score']) / self.config['recog_score']
        # if recog_score > 0:
        #     return 'unknown', min(1.0, 0.5 + recog_score)
        # return self.person_id_library[dist_sorted_idx[0][0]], min(1.0, 0.7 - recog_score)
        dist_mat = np.dot(features_array, self.person_feature_library.transpose())
        dist_sorted = np.max(dist_mat, axis=1)
        dist_sorted_idx = np.argmax(dist_mat, axis=1)
        if dist_sorted < self.config['recog_score']:
            return 'unknown', min(1.0, dist_sorted)
        return self.person_id_library[dist_sorted_idx[0]], min(1.0, dist_sorted)

    @staticmethod
    def draw_person_id_on_image(image, person_id, rect, method=1):
        """
        将人脸id和概率在原图像中画出
        Parameters
        ----------
        image 原图像
        person_id  人脸id
        rect 人脸矩形框[x1, y1,x2, y2, face_score]
        method rect为相对值或绝对值

        Returns
        -------

        """
        if method:
            width = image.shape[1]
            height = image.shape[0]
        else:
            width = 1.0
            height = 1.0
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        font = ImageFont.truetype('simkai.ttf', 15)
        fillColor = (127, 255, 0)
        position = (int(rect[0] * width), int(rect[1] * height + 10))
        draw = ImageDraw.Draw(pil_img)
        draw.text(position, person_id, font=font, fill=fillColor)
        image = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        return image
        # cv2.putText(image, f"{person_id}",
        #             (int(rect[0] * width), int(rect[1] * height+10)),
        #             1, 1, (255, 0, 255))

    def update_current_person(self, tracker_person_boxes, frame_num):
        """
        更新现有人物的轨迹状态
        Parameters
        ----------
        tracker_person_boxes 行人跟踪框
        frame_num 当前视频帧号

        Returns
        -------

        """
        new_current_persons = {}
        for person_box in tracker_person_boxes:
            init_current_persons = {'pedestrain_boxes': [],
                                    'face_boxes': [],
                                    'face_landmarks': [],
                                    'face_id': [],
                                    'face_score': [],
                                    'frame_num': []}
            person_id = int(person_box[-1])
            current_person = self.current_persons.get(person_id, init_current_persons)
            new_current_persons.update({person_id: current_person})
            new_current_persons[person_id]['pedestrain_boxes'].insert(0, person_box)
            new_current_persons[person_id]['frame_num'].insert(0, frame_num)

        self.current_persons = new_current_persons

    def crop_current_person(self):
        """轨迹保留最近的至多pic_nums张人脸得分大于0.5的信息"""
        for person_id in self.current_persons.keys():
            if len(self.current_persons[person_id]['frame_num']) > self.config['pic_nums']:
                indeces = np.where(np.array(self.current_persons[person_id]['face_score']) > self.config['recog_score'])
                indices = np.argsort(np.array(self.current_persons[person_id]['face_score']))[::-1]
                indices = [index for index in indices[:self.config['pic_nums']] if index in indeces[0]]
                for key in self.current_persons[person_id].keys():
                    if key == 'name':
                        continue
                    self.current_persons[person_id][key] = \
                        [self.current_persons[person_id][key][index] for index in indices]

    def get_current_face_id(self):
        """采用轨迹中人脸得分大于0.5部分的id众数作为当前轨迹的默认id"""
        for person_id in self.current_persons.keys():
            indeces = np.where(np.array(self.current_persons[person_id]['face_score']) > self.config['recog_score'])
            indices = np.argsort(np.array(self.current_persons[person_id]['face_score']))[::-1]
            face_id_list = [self.current_persons[person_id]['face_id'][index] for index in indices if
                            index in indeces[0]]
            if len(face_id_list) >= 1:
                counts = Counter(face_id_list)
                self.current_persons[person_id].update({'name': counts.most_common(1)[0][0]})

    def detect(self, video):
        """
        对单个视频的包含行人检测+行人跟踪+人脸检测+人脸质量过滤+人脸矫正+人脸识别的整套流程
        Parameters
        ----------
        video      cv2的VideoCapture()类

        Returns
        -------
            无
        """
        frame_num = 0
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.set_config(width, height)
        cv2.namedWindow("petition", 0)
        while True:
            _, frame = video.read()
            if frame is None:
                break
            frame_num += 1
            start_time = time.time()
            image = frame.copy()
            if frame_num % 1 == 0:
                person_boxes = self.pedestrain_detector.detect(image)
            else:
                person_boxes = []
            # print(person_boxes)
            tracker_person_boxes = self.tracker.track(_, np.array(person_boxes))
            self.update_current_person(tracker_person_boxes, frame_num)
            # 将当前帧的行人检测结果和人脸检测结果写入self.current_persons
            for person_id, current_person in self.current_persons.items():
                person_box = current_person['pedestrain_boxes'][0]
                crop_person_box = self.crop_pedestrain_box(person_box)
                person_image = self.crop_pedestrain_image(frame, crop_person_box, method=1)
                # cv2.imshow("person", person_image)
                # face_box = self.face_detector.detect(person_image)
                face_box, detect_boxes_landmark = self.face_detector.detect_boxes_landmarks(person_image)
                # print(frame_num, face_box, detect_boxes_landmark)
                if face_box.shape[0]:
                    face_box, detect_boxes_landmark = self.face_box_transform(crop_person_box, face_box,
                                                                              detect_boxes_landmark)
                    face_box = face_box[0, :]
                    detect_boxes_landmark = detect_boxes_landmark[0, :]
                    self.current_persons[person_id]['face_boxes'].insert(0, face_box)
                    self.current_persons[person_id]['face_landmarks'].insert(0, detect_boxes_landmark)
                else:
                    self.current_persons[person_id]['face_boxes'].insert(0, [])
                    self.current_persons[person_id]['face_landmarks'].insert(0, [])
            current_pedestrain_boxes = []
            current_face_boxes = []
            current_face_landmarks = []
            for person_id, current_person in self.current_persons.items():  # bug
                pedestrain_box = current_person['pedestrain_boxes'][0]
                face_box = current_person['face_boxes'][0]
                face_landmark = current_person['face_landmarks'][0]
                # 当前行人框没有检测到人脸跳过进行下一行人轨迹的人脸检测
                if not len(face_box):
                    self.current_persons[person_id]['face_id'].insert(0, 'Unknown')
                    self.current_persons[person_id]['face_score'].insert(0, 0)
                    continue
                # 由当前行人框裁剪出需要的人脸框并进行人脸关键点的矫正
                current_pedestrain_boxes.append(pedestrain_box)
                crop_face_box = self.enlarge_box(face_box.copy(), w_scale=1.5, h_scale=1.5)
                face_image = self.crop_face_image(frame, crop_face_box, method=1)
                crop_face_landmark = self.face_lankmark_transform(crop_face_box, face_landmark.copy(), method=0)
                face_image = self.face_align.align(face_image, crop_face_landmark.reshape(5, 2))
                face_image = cv2.resize(face_image, (112, 112))
                # 设置人脸识别频率, 不识别时继承当前轨迹中的'name'结果
                if frame_num % 5 != 0:
                    self.get_current_face_id()
                    face_id = self.current_persons[person_id].get('name', 'Unknown')
                    self.current_persons[person_id]['face_id'].insert(0, face_id)
                    self.current_persons[person_id]['face_score'].insert(0, 0)
                    face_box[4] = 0
                    # 只显示有人脸的结果
                    cv2.imshow("faces", face_image)
                    image = self.draw_person_id_on_image(image, face_id, face_box)
                    current_face_boxes.append(face_box)
                    current_face_landmarks.append(face_landmark)
                else:
                    # 人脸过滤
                    if self.face_boxes_filter(face_image, crop_face_box):
                        # 此处进行人脸识别
                        face_id, dist = self.recognition(face_image)
                        face_box[4] = dist
                        self.current_persons[person_id]['face_id'].insert(0, face_id)
                        self.current_persons[person_id]['face_score'].insert(0, face_box[4])

                        cv2.imshow("faces", face_image)
                        image = self.draw_person_id_on_image(image, face_id, face_box)
                        current_face_boxes.append(face_box)
                        current_face_landmarks.append(face_landmark)
                    else:
                        self.current_persons[person_id]['face_id'].insert(0, 'Unknown')
                        self.current_persons[person_id]['face_score'].insert(0, 0)
            self.crop_current_person()
            print(self.current_persons)
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            draw_track_rects(image, np.array(tracker_person_boxes), method=1)
            draw_detection_rects(image, np.array(current_face_boxes), np.array(current_face_landmarks),
                                 color=(99, 99, 238), method=1)
            cv2.putText(image, f"fps: {fps:.01f}", (50, 50), 1, 2, (0, 255, 0), 2)
            if "win" in sys.platform:
                cv2.imshow("petition", image)
                key = cv2.waitKey(self.decay_time)
                if key == 32:
                    self.auto_play_flag = not self.auto_play_flag
                    self.decay_time = 1 if self.auto_play_flag else 0
                if key == 27:
                    break
                if key == ord('l'):
                    frame_num += 100
                    frame_num = min(frame_num, video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    def destroy(self):
        self.pedestrain_detector.destroy()


def main():
    person_library = r"F:\tmp\person_search\face_imgs"
    work_root = os.path.dirname(os.path.dirname(working_root))
    # video_path = os.path.join(work_root, "data/video_data/videos/1.mp4")
    # video_path = os.path.join(work_root, "data/video_data/videos/inside-alg-000.mp4")
    video_path = r"E:\videos\outside-data.mp4"
    onnx_file_path = os.path.join(work_root, r"checkpoints/face_reid/backbone_ir50_asia-sim.onnx")
    # onnx_file_path = os.path.join(work_root, r"checkpoints/face_reid/plr_osnet_246_2.1345-sim.onnx")
    face_detector = FaceDetector(r"D:\workspace\Pytorch\bitbucket\face_projects\thirdparty"
                                 r"\face_detect_inference\face_detect_landmark_onnx\onnx\rtface_mb_256_sim.onnx")
    # face_detector.set_config('width', 128)
    # face_detector.set_config('height', 128)
    # face_detector = OnnxObjectDetector(r"D:\workspace\Pytorch\bitbucket\face_projects\thirdparty"
    #                                    r"\face_detect_inference\face_detect_onnx\onnx_model"
    #                                    r"\mobilenet_v2_0.25_43_0.1162-sim.onnx")
    extractor = FeatureExtract(onnx_file_path)
    extractor.set_config('mean', [127.5, 127.5, 127.5])
    extractor.set_config('stddev', [128, 128, 128])
    extractor.set_config('divisor', 1.0)
    person_search = VideoRecognition(person_library, face_detector=face_detector, extractor=extractor)
    person_search.person_library_feature_extract()
    print(person_search.person_feature_library.shape)
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video: ", video_path)
        return
    person_search.detect(video)


if __name__ == '__main__':
    main()
