"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from scipy import interpolate
# plt.switch_backend('agg')
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})


# Support: ['calculate_roc', 'calculate_accuracy', 'calculate_val', 'calculate_val_far', 'evaluate']


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
        # dist = pdist(np.vstack([embeddings1, embeddings2]), 'cosine')

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca > 0:
            print("doing pca on", fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        #         print('best_threshold_index', best_threshold_index, acc_train[best_threshold_index])
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    '''
    Copy from [insightface](https://github.com/deepinsight/insightface)
    :param thresholds:
    :param embeddings1:
    :param embeddings2:
    :param actual_issame:
    :param far_target:
    :param nrof_folds:
    :return:
    '''
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame),
                                                        nrof_folds=nrof_folds, pca=pca)
    #     thresholds = np.arange(0, 4, 0.001)
    #     val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
    #                                       np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    #     return tpr, fpr, accuracy, best_thresholds, val, val_std, far
    return tpr, fpr, accuracy, best_thresholds


def load_bin(path):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    # print(issame_list)
    issame_array = np.array(issame_list, dtype=np.int8())
    # print(np.where(issame_array == 1)[0].shape)
    # print(len(issame_list))
    # print(len(bins))
    return issame_array


class TorchInference(object):
    def __init__(self, model_path=None, device=None):
        """
        对TorchInference进行初始化

        Parameters
        ----------
        model_path : str
            pytorch模型的路径，推荐使用绝对路径
        """
        super().__init__()
        self.model_path = model_path
        self.device = device
        if self.model_path is None:
            print("please set pytorch model path!\n")
            exit(-1)
        self.session = None
        self.model_loader()

    def model_loader(self):
        if torch.__version__ < "1.0.0":
            print("Pytorch version is not  1.0.0, please check it!")
            exit(-1)
        if self.model_path is None:
            print("Please set model path!!")
            exit(-1)
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # check_point = torch.load(self.checkpoint_file_path, map_location=self.device)
        # self.model = check_point['net'].to(self.device)
        self.session = torch.jit.load(self.model_path, map_location=self.device)
        # 如果模型为pytorch0.3.1版本，需要以下代码添加BN内的参数
        # for _, module in self.model._modules.items():
        #     recursion_change_bn(self.model)
        self.session.eval()

    def inference(self, x: torch.Tensor):
        """
        pytorch的推理
        Parameters
        ----------
        x : torch.Tensor
            pytorch模型输入

        Returns
        -------
        torch.Tensor
            pytorch模型推理结果
        """
        x = x.to(self.device)
        self.session = self.session.to(self.device)
        outputs = self.session(x)
        return outputs


class Evaluation(object):
    def __init__(self, root='', pairs_file=None, batch_inference=None):
        self.embeddings = None
        self.actual_issame = list()
        self.image_list = list()
        self.root = root
        self.pairs_file = pairs_file
        self.pairs_bin_file = os.path.join(os.path.dirname(self.pairs_file), "lfw.npy")
        self.batch_inference = batch_inference
        self.config = {
            'width': 112,
            'height': 112,
            'color_format': 'RGB',
            'mean': [0.5, 0.5, 0.5],
            'stddev': [0.5, 0.5, 0.5],
            'divisor': 255.0,
            "feature_size": 512,
            "batch_size": 32,
            "hflip": True
        }

    def set_config(self, key: str, value):
        if key not in self.config:
            print("config key error! please check it!")
            exit()
        self.config[key] = value

    def generate_actual_issame(self):
        pass

    def get_actual_issame(self):
        pairs_file = open(self.pairs_file, "r")
        pairs = pairs_file.readlines()[1:]
        for index in range(len(pairs)):
            pairs_info = pairs[index].strip().split()
            if len(pairs_info) == 3:
                cur_person_name = pairs_info[0]
                image_index_1 = int(pairs_info[1]) - 1
                image_index_2 = int(pairs_info[2]) - 1
                cur_person_folder = os.path.join(self.root, cur_person_name)
                cur_person_name_list = os.listdir(cur_person_folder)
                image_path1 = os.path.join(cur_person_folder, cur_person_name_list[image_index_1])
                image_path2 = os.path.join(cur_person_folder, cur_person_name_list[image_index_2])
                image1 = cv2.imread(image_path1)
                image2 = cv2.imread(image_path2)
                try:
                    self.image_list.extend([self.resize_and_center_crop(image1), self.resize_and_center_crop(image2)])
                except:
                    print(image_path1, image_path2)
                    exit(-1)
                self.actual_issame.append(True)
            elif len(pairs_info) == 4:
                cur_person_name1 = pairs_info[0]
                image_index_1 = int(pairs_info[1]) - 1
                cur_person_name2 = pairs_info[2]
                image_index_2 = int(pairs_info[3]) - 1
                cur_person_folder1 = os.path.join(self.root, cur_person_name1)
                cur_person_name_list1 = os.listdir(cur_person_folder1)
                image_path1 = os.path.join(cur_person_folder1, cur_person_name_list1[image_index_1])
                cur_person_folder2 = os.path.join(self.root, cur_person_name2)
                cur_person_name_list2 = os.listdir(cur_person_folder2)
                image_path2 = os.path.join(cur_person_folder2, cur_person_name_list2[image_index_2])
                image1 = cv2.imread(image_path1)
                image2 = cv2.imread(image_path2)
                try:
                    self.image_list.extend([self.resize_and_center_crop(image1), self.resize_and_center_crop(image2)])
                except:
                    print(image_path1, image_path2)
                    exit(-1)
                self.actual_issame.append(False)
            else:
                print(f"line {index} sparse error! continue next one...")
                pass

    @staticmethod
    def normalize(nparray, order=2, axis=0):
        """Normalize a N-D numpy array along the specified axis."""
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)

    @staticmethod
    def l2_norm(input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)
        return output

    def resize_and_center_crop(self, image: np.ndarray):
        resize_image = cv2.resize(image, (128, 128))
        crop_image = self.center_crop(resize_image, self.config['width'], self.config['height'])
        return crop_image

    @staticmethod
    def center_crop(image: np.ndarray, width: int, height: int):
        assert (image.shape[1] >= width) and (image.shape[0] >= height)
        crop_x = int(image.shape[1] - width) // 2
        crop_y = int(image.shape[0] - height) // 2
        crop_image = image[crop_y:crop_y + height, crop_x:crop_x + width, :]
        return crop_image

    def _pre_process(self, image: np.ndarray, hflip=False) -> np.ndarray:
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
        if self.config['color_format'] == "RGB":
            image = image[:, :, :, ::-1]
        if hflip:
            image = image[:, ::-1, :, :]
        # if self.config['width'] > 0 and self.config['height'] > 0:
        #     # image = cv2.resize(image, (self.config['width'], self.config['height']))
        #     image = cv2.resize(image, (128, 128))
        #     image = self.center_crop(image, self.config['width'], self.config['height'])
        input_image = (np.array(image, dtype=np.float32) / self.config['divisor'] - self.config['mean']) / self.config[
            'stddev']
        return input_image

    def _post_process(self, features):
        features = self.l2_norm(features)
        features = torch.squeeze(features)
        features_array = features.data.cpu().numpy()
        # features_array = self.normalize(features_array, axis=1)
        return features_array

    def batch_feature_extract(self, images):
        batch_images = self._pre_process(images)
        batch_images = np.transpose(batch_images, [0, 3, 1, 2])
        batch_images = torch.from_numpy(batch_images).float()
        # features = model(batch_images)
        output = self.batch_inference(batch_images)
        if self.config['hflip']:
            hflip_batch_images = self._pre_process(images, hflip=True)
            hflip_batch_images = np.transpose(hflip_batch_images, [0, 3, 1, 2])
            hflip_batch_images = torch.from_numpy(hflip_batch_images).float()
            # features = model(batch_images)
            output += self.batch_inference(hflip_batch_images)
        features_array = self._post_process(output)
        return features_array

    def feature_extract(self):
        person_batch = len(self.image_list) // self.config['batch_size']
        person_id_features = np.zeros([0, self.config['feature_size']], np.float)
        print("start feature extract...")
        for index in range(person_batch):
            if index % (person_batch // 10) == 0:
                print(f"process {index}/{person_batch}...")
            batch_images = np.stack(self.image_list[index * self.config['batch_size']:
                                                    (index + 1) * self.config['batch_size']])
            features_array = self.batch_feature_extract(batch_images)
            person_id_features = np.vstack((person_id_features, features_array))
        if len(self.image_list) % self.config['batch_size']:
            batch_images = np.stack(self.image_list[person_batch * self.config['batch_size']::])
            features_array = self.batch_feature_extract(batch_images)
            person_id_features = np.vstack((person_id_features, features_array))
        self.embeddings = person_id_features

    def evaluate(self):
        if os.path.exists(self.pairs_bin_file):
            self.image_list, self.actual_issame = np.load(self.pairs_bin_file, allow_pickle=True)
        elif self.pairs_file is None:
            self.generate_actual_issame()
            np.save(self.pairs_bin_file, [self.image_list, self.actual_issame])
        else:
            self.get_actual_issame()
            np.save(self.pairs_bin_file, [self.image_list, self.actual_issame])
        self.feature_extract()
        tpr, fpr, accuracy, best_thresholds = evaluate(self.embeddings, self.actual_issame)
        return tpr, fpr, accuracy, best_thresholds


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    plt.show()


if __name__ == '__main__':
    root = r"F:\Database\face\lfw\lfw-deepfunneled\lfw-deepfunneled"
    pair_file_path = r"F:\Database\face\lfw\test\pairs.txt"
    root_working = os.path.split(os.path.realpath(__file__))[0]
    model_path = os.path.join(os.path.dirname(os.path.dirname(root_working)),
                              # r"checkpoints/face_reid/plr_osnet_246_2.1345_jit.pth")
                              # r"checkpoints/face_reid/backbone_ir50_ms1m_epoch120_jit.pth")
                              # r"checkpoints/face_reid/model_mobilefacenet_jit.pth")
                              r"checkpoints/face_reid/backbone_ir50_asia_jit.pth")
    torch_inference = TorchInference(model_path)
    print("load model to inference success!")
    evalution = Evaluation(root, pair_file_path, torch_inference.inference)
    # evalution.set_config('mean', [0.4914, 0.4822, 0.4465])
    # evalution.set_config('stddev',  [0.247, 0.243, 0.261])
    # evalution.set_config('divisor', 255.0)
    # evalution.set_config('feature_size', 512)
    tpr, fpr, accuracy, best_thresholds = evalution.evaluate()
    print(tpr, fpr, accuracy, best_thresholds)
    gen_plot(fpr, tpr)
