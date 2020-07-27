import os
import cv2
import sys
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
working_root = os.path.dirname(os.path.dirname(os.path.split(os.path.realpath(__file__))[0]))
sys.path.append(working_root)
from modules.face_recognition import (PyFeatureExtract, OnnxFeatureExtract)


def main():

    onnx_file_path = os.path.join(working_root, r"checkpoints/face_reid/backbone_ir50_asia-sim.onnx")
    # onnx_file_path = os.path.join(work_root, r"checkpoints/face_reid/plr_osnet_246_2.1345-sim.onnx")
    extractor = OnnxFeatureExtract(onnx_file_path)
    extractor.set_config('mean', [127.5, 127.5, 127.5])
    extractor.set_config('stddev', [128.0, 128.0, 128.0])
    extractor.set_config('divisor', 1.0)

    image_path = r"face1.npy"
    image = np.load(image_path)
    image = np.expand_dims(image, 0)
    feature = extractor.extractor.inference(image)[0]

    # image_path = r"images\face1.jpg"
    # image = cv2.imread(image_path)
    # feature = extractor.feature_extract(image)

    print(extractor.extractor.config)
    print(feature)
    print(feature.shape)


if __name__ == '__main__':
    main()
