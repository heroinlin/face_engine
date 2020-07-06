import os
import glob
import cv2
import numpy as np
working_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.dirname(os.path.dirname(working_root)))
from modules.face_recognition import (PyFeatureExtract, OnnxFeatureExtract)


class PersonSearch():
    def __init__(self, person_library):
        self.person_library = person_library
        self.person_id_library = None
        self.person_feature_library = None
        self.feature_extractor = OnnxFeatureExtract()
        self.config = {'feature_size': 512}
    
    def person_library_feature_extract(self):
        self.person_feature_library = np.zeros([0, self.config['feature_size']], np.float)
        self.person_id_library = sorted(os.listdir(self.person_library))
        for person_folder in self.person_id_library:
            person_id_features = np.zeros([0, self.config['feature_size']], np.float)
            for image_path in sorted(glob.glob(os.path.join(self.person_library, person_folder, "*.jpg"))):
                image = cv2.imread(image_path)
                features_array = self.feature_extractor.feature_extract(image)
                person_id_features = np.vstack((person_id_features, features_array))
            # 取特征的均值
            person_id_features = np.mean(person_id_features, 0)
            # 归一化
            sq_sum = 1 / np.sqrt(np.sum(np.square(person_id_features)) + 1e-6)
            sq_sum = np.expand_dims(sq_sum, 1)
            person_id_features = person_id_features * sq_sum
            self.person_feature_library = np.vstack((self.person_feature_library, person_id_features))
    
    def validation(self, image):
        features_array = self.feature_extractor.feature_extract(image)
        if self.person_feature_library is None:
            print("No person in the person library!")
            exit(-1)
        dist_mat = 1 - np.dot(features_array, self.person_feature_library.transpose())
        dist_sorted = np.sort(dist_mat, axis=1)
        dist_sorted_idx = np.argsort(dist_mat, axis=1)
        return self.person_id_library[dist_sorted_idx[0]], dist_sorted[0]


def main():
    person_library = "./person_library"
    query_image_path = "unknown.jpg"
    person_search = PersonSearch(person_library)
    person_search.person_library_feature_extract()
    query_image = cv2.imread(query_image_path)
    person_id , dist = person_search.validation(query_image)
    print(person_id , dist)


if __name__ == '__main__':
    main()
    