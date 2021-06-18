# -*- coding: utf-8 -*-
import argparse
import glob
import os
import sys
import time
from PIL import Image
import cv2
import numpy as np
working_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.dirname(os.path.dirname(working_root)))
from modules.object_detection import OnnxObjectDetector as ObjectDetector
# from modules.object_detection import PyObjectDetector as ObjectDetector
from modules.utils import compute_time


def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image)
    return image


def main():
    detector = ObjectDetector()
    image_path = os.path.join(os.getcwd(), "./data/image_data/images/00000_1.jpg")
    image = cv2.imread(image_path, 1)
    input_image = detector.detector._pre_process(image)
    pred = detector.detector.inference(input_image)
    outputs = detector.detector._post_process(pred)
    file = load_image(image_path)
    cv_load_image_time = compute_time(cv2.imread, [image_path])
    print("avg opencv load image time is {:02f} ms".format(cv_load_image_time))

    PIL_load_image_time = compute_time(load_image, [image_path])
    print("avg PIL load image time is {:02f} ms".format(PIL_load_image_time))

    preprocess_time = compute_time(detector.detector._pre_process, [image])
    print("avg preprocess time is {:02f} ms".format(preprocess_time))

    inference_time = compute_time(detector.detector.inference, [input_image])
    print("avg inference time is {:02f} ms".format(inference_time))

    postprocess_time = compute_time(detector.detector._post_process, [pred])
    print("avg postprocess time is {:02f} ms".format(postprocess_time))

    total_time = compute_time(detector.detect, [image])
    print("avg total predict time is {:02f} ms".format(total_time))


if __name__ == '__main__':
    main()
