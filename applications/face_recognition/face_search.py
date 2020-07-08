import os
import sys
import glob
import cv2
import numpy as np
working_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.dirname(os.path.dirname(working_root)))
from modules.face_recognition import (PyFeatureExtract, OnnxFeatureExtract)


class PersonSearch():
    def __init__(self, person_library, onnx_file_path):
        self.person_library = person_library
        self.person_id_library = None
        self.person_feature_library = None
        self.feature_extractor = OnnxFeatureExtract(onnx_file_path)
        self.config = {'feature_size': 512}

    @staticmethod
    def normalize(nparray, order=2, axis=0):
        """Normalize a N-D numpy array along the specified axis."""
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)

    def person_library_feature_extract(self):
        self.person_feature_library = np.zeros([0, self.config['feature_size']], np.float)
        self.person_id_library = sorted(os.listdir(self.person_library))
        print(self.person_id_library)
        for person_folder in self.person_id_library:
            person_id_features = np.zeros([0, self.config['feature_size']], np.float)
            for image_path in sorted(glob.glob(os.path.join(self.person_library, person_folder, "*.[jp][pn]g"))):
                image = cv2.imdecode(np.fromfile(image_path, np.uint8()), 1)
                features_array = self.feature_extractor.feature_extract(image)
                features_array = features_array.reshape(-1, self.config['feature_size'])
                person_id_features = np.vstack((person_id_features, features_array))
            # 取特征的均值
            if person_id_features.shape[0] > 1:
                person_id_features = np.mean(person_id_features, 0, keepdims=True)
            # 归一化
            person_id_features = self.normalize(person_id_features, axis=1)
            self.person_feature_library = np.vstack((self.person_feature_library, person_id_features))
    
    def validation(self, image_path):
        image = cv2.imdecode(np.fromfile(image_path, np.uint8()), 1)
        features_array = self.feature_extractor.feature_extract(image)
        features_array = features_array.reshape(-1, self.config['feature_size'])
        features_array = self.normalize(features_array, axis=1)
        if self.person_feature_library is None:
            print("No person in the person library!")
            exit(-1)
        dist_mat = 1 - np.dot(features_array, self.person_feature_library.transpose())
        dist_sorted = np.sort(dist_mat, axis=1)
        dist_sorted_idx = np.argsort(dist_mat, axis=1)
        print(dist_mat)
        return self.person_id_library[dist_sorted_idx[0][0]], dist_sorted[0][0]


def main():
    person_library = r"F:\tmp\person_search\worker_face.door"
    work_root = os.path.dirname(os.path.dirname(working_root))
    # onnx_file_path = os.path.join(work_root, r"checkpoints/face_reid/backbone_ir50_asia-sim.onnx")
    onnx_file_path = os.path.join(work_root, r"checkpoints/face_reid/plr_osnet_237_2.1969-sim.onnx")
    person_search = PersonSearch(person_library, onnx_file_path)
    person_search.person_library_feature_extract()
    query_image_folder = r"F:\tmp\person_search\worker_face.inside"
    for query_image_path in sorted(glob.glob(query_image_folder + "/*.[jp][pn]g")):
        person_id, dist = person_search.validation(query_image_path)
        print(os.path.basename(query_image_path), person_id, dist)


if __name__ == '__main__':
    main()
