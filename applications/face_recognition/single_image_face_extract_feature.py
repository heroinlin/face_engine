# -*- coding: utf-8 -*-

import numpy as np
import onnxruntime

import os
import cv2
from skimage import transform as trans

working_root = os.path.split(os.path.realpath(__file__))[0]


class FaceWarpException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
            __file__, super.__str__(self))


class PtFaceAlign(object):
    def __init__(self):
        self.config = {
            'width': 112,
            'height': 112,
            'align_type': 'smilarity',
            'reference_facial_points': [[38.29459953, 51.69630051],
                                        [73.53179932, 51.50139999],
                                        [56.02519989, 71.73660278],
                                        [41.54930115, 92.3655014],
                                        [70.72990036, 92.20410156]]
        }

    def get_reference_facial_points(self, output_size=None,
                                    inner_padding_factor=0.0,
                                    outer_padding=(0, 0),
                                    default_square=False):
        tmp_5pts = np.array(self.config['reference_facial_points'])
        tmp_crop_size = np.array([self.config['width'], self.config['height']])

        # 0) make the inner region a square
        if default_square:
            size_diff = max(tmp_crop_size) - tmp_crop_size
            tmp_5pts += size_diff / 2
            tmp_crop_size += size_diff

        # print('---> default:')
        # print('              crop_size = ', tmp_crop_size)
        # print('              reference_5pts = ', tmp_5pts)

        if (output_size and
                output_size[0] == tmp_crop_size[0] and
                output_size[1] == tmp_crop_size[1]):
            # print('output_size == DEFAULT_CROP_SIZE {}: return default reference points'.format(tmp_crop_size))
            return tmp_5pts

        if (inner_padding_factor == 0 and
                outer_padding == (0, 0)):
            if output_size is None:
                print('No paddings to do: return default reference points')
                return tmp_5pts
            else:
                raise FaceWarpException(
                    'No paddings to do, output_size must be None or {}'.format(tmp_crop_size))

        # check output size
        if not (0 <= inner_padding_factor <= 1.0):
            raise FaceWarpException('Not (0 <= inner_padding_factor <= 1.0)')

        if ((inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0)
                and output_size is None):
            output_size = tmp_crop_size * \
                          (1 + inner_padding_factor * 2).astype(np.int32)
            output_size += np.array(outer_padding)
            print('              deduced from paddings, output_size = ', output_size)

        if not (outer_padding[0] < output_size[0]
                and outer_padding[1] < output_size[1]):
            raise FaceWarpException('Not (outer_padding[0] < output_size[0]'
                                    'and outer_padding[1] < output_size[1])')

        # 1) pad the inner region according inner_padding_factor
        # print('---> STEP1: pad the inner region according inner_padding_factor')
        if inner_padding_factor > 0:
            size_diff = tmp_crop_size * inner_padding_factor * 2
            tmp_5pts += size_diff / 2
            tmp_crop_size += np.round(size_diff).astype(np.int32)

        # print('              crop_size = ', tmp_crop_size)
        # print('              reference_5pts = ', tmp_5pts)

        # 2) resize the padded inner region
        # print('---> STEP2: resize the padded inner region')
        size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2
        # print('              crop_size = ', tmp_crop_size)
        # print('              size_bf_outer_pad = ', size_bf_outer_pad)

        if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]:
            raise FaceWarpException('Must have (output_size - outer_padding)'
                                    '= some_scale * (crop_size * (1.0 + inner_padding_factor)')

        scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
        # print('              resize scale_factor = ', scale_factor)
        tmp_5pts = tmp_5pts * scale_factor
        #    size_diff = tmp_crop_size * (scale_factor - min(scale_factor))
        #    tmp_5pts = tmp_5pts + size_diff / 2
        tmp_crop_size = size_bf_outer_pad
        # print('              crop_size = ', tmp_crop_size)
        # print('              reference_5pts = ', tmp_5pts)

        # 3) add outer_padding to make output_size
        reference_5point = tmp_5pts + np.array(outer_padding)
        tmp_crop_size = output_size
        # print('---> STEP3: add outer_padding to make output_size')
        # print('              crop_size = ', tmp_crop_size)
        # print('              reference_5pts = ', tmp_5pts)
        #
        # print('===> end get_reference_facial_points\n')

        return reference_5point

    @staticmethod
    def get_affine_transform_matrix(src_pts, dst_pts):
        tfm = np.float32([[1, 0, 0], [0, 1, 0]])
        n_pts = src_pts.shape[0]
        ones = np.ones((n_pts, 1), src_pts.dtype)
        src_pts_ = np.hstack([src_pts, ones])
        dst_pts_ = np.hstack([dst_pts, ones])

        A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

        if rank == 3:
            tfm = np.float32([
                [A[0, 0], A[1, 0], A[2, 0]],
                [A[0, 1], A[1, 1], A[2, 1]]
            ])
        elif rank == 2:
            tfm = np.float32([
                [A[0, 0], A[1, 0], 0],
                [A[0, 1], A[1, 1], 0]
            ])

        return tfm

    # BGR
    def warp_and_crop_face(self, src_img,
                           facial_pts,
                           reference_pts=None,
                           crop_size=(112, 112),
                           align_type='smilarity'):

        if reference_pts is None:
            if crop_size[0] == 112 and crop_size[1] == 112:
                reference_pts = self.config['reference_facial_points']
            else:
                default_square = False
                inner_padding_factor = 0
                outer_padding = (0, 0)
                output_size = crop_size

                reference_pts = self.get_reference_facial_points(output_size,
                                                                 inner_padding_factor,
                                                                 outer_padding,
                                                                 default_square)

        ref_pts = np.float32(reference_pts)
        ref_pts_shp = ref_pts.shape
        if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
            raise FaceWarpException(
                'reference_pts.shape must be (K,2) or (2,K) and K>2')

        if ref_pts_shp[0] == 2:
            ref_pts = ref_pts.T

        src_pts = np.float32(facial_pts)
        src_pts_shp = src_pts.shape
        if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
            raise FaceWarpException(
                'facial_pts.shape must be (K,2) or (2,K) and K>2')

        if src_pts_shp[0] == 2:
            src_pts = src_pts.T

        if src_pts.shape != ref_pts.shape:
            raise FaceWarpException(
                'facial_pts and reference_pts must have the same shape')

        if align_type is 'cv2_affine':
            tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
        #        print('cv2.getAffineTransform() returns tfm=\n' + str(tfm))
        elif align_type is 'affine':
            tfm = self.get_affine_transform_matrix(src_pts, ref_pts)
        #        print('get_affine_transform_matrix() returns tfm=\n' + str(tfm))
        else:
            # tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)
            tform = trans.SimilarityTransform()
            tform.estimate(src_pts, ref_pts)
            tfm = tform.params[0:2, :]

        face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

        return face_img  # BGR

    def align(self, image: np.ndarray, facial_pts: np.ndarray) -> list:
        return self.warp_and_crop_face(image,
                                       facial_pts,
                                       self.config['reference_facial_points'],
                                       (self.config['width'], self.config['height']),
                                       self.config['align_type'])

    def set_config(self, key: str, value):
        self.config[key] = value

    def destroy(self):
        pass


class ONNXInference(object):
    def __init__(self, model_path=None):
        """
        对ONNXInference进行初始化

        Parameters
        ----------
        model_path : str
            onnx模型的路径，推荐使用绝对路径
        """
        super().__init__()
        self.model_path = model_path
        if self.model_path is None:
            print("please set onnx model path!\n")
            exit(-1)
        self.session = onnxruntime.InferenceSession(self.model_path)

    def inference(self, x: np.ndarray):
        """
        onnx的推理
        Parameters
        ----------
        x : np.ndarray
            onnx模型输入

        Returns
        -------
        np.ndarray
            onnx模型推理结果
        """
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        outputs = self.session.run(output_names=[output_name],
                                   input_feed={input_name: x.astype(np.float32)})
        return outputs


class FeatureExtract(ONNXInference):
    def __init__(self, model_path=None):
        """对FeatureExtract进行初始化

        Parameters
        ----------
        model_path : str
            onnx模型的路径，推荐使用绝对路径
        """
        if model_path is None:
            model_path = os.path.join(working_root,
                                      'onnx',
                                      "backbone_ir50_asia-sim.onnx")
        super(FeatureExtract, self).__init__(model_path)
        self.face_align = PtFaceAlign()
        self.config = {
            'width': 112,
            'height': 112,
            'color_format': 'RGB',
            'mean': [127.5, 127.5, 127.5],
            'stddev': [128, 128, 128],
            'divisor': 1.0,
        }

    def set_config(self, key: str, value):
        if key not in self.config:
            print("config key error! please check it!")
            exit()
        self.config[key] = value

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

    def face_lankmark_transform(self, face_box, landmark, width, height):
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
        landmark[::2] = (landmark[::2] - face_box[0]) * width
        landmark[1::2] = (landmark[1::2] - face_box[1]) * height
        landmark = landmark.astype(np.int8)
        return landmark

    def _pre_process(self, image: np.ndarray, box: np.ndarray, landmark: np.ndarray) -> np.ndarray:
        """对图像进行预处理

        Parameters
        ----------
        image : np.ndarray
            输入的原始图像，BGR格式，通常使用cv2.imread读取得到

        Returns
        -------
        np.ndarray
            原始图像经过预处理后得到的数组
        """
        box = self.enlarge_box(box, w_scale=1.5, h_scale=1.5)
        crop_image = self.crop_face_image(image, box, method=1)
        landmark = self.face_lankmark_transform(box, landmark, image.shape[1], image.shape[0])
        image = self.face_align.align(crop_image, landmark.reshape(5, 2))
        if self.config['color_format'] == "RGB":
            image = image[:, :, ::-1]
        if self.config['width'] > 0 and self.config['height'] > 0:
            image = cv2.resize(image, (self.config['width'], self.config['height']))
        input_image = (np.array(image, dtype=np.float32) / self.config['divisor'] - self.config['mean']) / self.config[
            'stddev']
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, 0)
        return input_image

    def _post_process(self, feature):
        feature = self.normalize(feature, axis=1)
        return feature

    def feature_extract(self, image: np.ndarray, box: np.ndarray,  landmark: np.ndarray) -> np.ndarray:
        """对输入图像提取特征

        Parameters
        ----------
        image : np.ndarray
            输入图片，BGR格式，通常使用cv2.imread获取得到

        Returns
        -------
        np.ndarray
            返回特征
        """
        src_image = self._pre_process(image.copy(), box, landmark)
        feature = self.inference(src_image)[0]
        feature = self._post_process(feature)
        return np.array(feature)


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


if __name__ == '__main__':
    work_root = os.path.dirname(os.path.dirname(working_root))
    onnx_file_path = os.path.join(work_root, r"checkpoints/face_reid/backbone_ir50_asia-sim.onnx")
    extractor = FeatureExtract(onnx_file_path)
    image_path = r"1.jpg"
    image = cv2.imread(image_path)

    # box = np.array([989, 387, 1058, 467])
    box = np.array([0.515, 0.358, 0.551, 0.432])
    # landmark = np.array([1011, 10, 1038, 408, 1029, 422, 1021, 442, 1038, 442], np.float32())
    landmark = np.array([0.527, 0.380, 0.541, 0.378, 0.536, 0.391, 0.532, 0.410, 0.541, 0.410], np.float32())
    feature = extractor.feature_extract(image.copy(), box.copy(), landmark.copy())
    # print(feature)
    draw_landmarks(image, landmark.reshape([5, 2]))
    cv2.imshow("image", image)
    cv2.waitKey()
