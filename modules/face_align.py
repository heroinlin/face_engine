# -*- coding: utf-8 -*-

import abc
import os
import sys
import cv2
import numpy as np
from skimage import transform as trans


class IFaceAlign(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def align(self, image: np.ndarray, facial_pts: np.ndarray):
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


class FaceWarpException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
            __file__, super.__str__(self))


class PtFaceAlign(IFaceAlign):
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
