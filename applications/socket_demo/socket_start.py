#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a face detection model."""
import os
import sys
import torch
import cv2
import numpy as np
import time
import socket
import struct
import json
import argparse
working_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.dirname(os.path.dirname(working_root)))
from modules.object_detection import PyObjectDetector as ObjectDetector
np.set_printoptions(precision=3, suppress=True)


def socket_start(port=6066):
    demo = ObjectDetector()
    print("load model success!\n")
    fdSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    fdSocket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                        struct.pack('ii', 1, 0))
    fdSocket.bind(('127.0.0.1', port))
    fdSocket.listen(10)
    while True:
        conn, addr = fdSocket.accept()
        data, _ = conn.recvfrom(1024)
        data = data.decode('utf-8')
        start_time = time.time()
        image = cv2.imread(data)
        detect_msg = ""
        if image is None:
            detect_msg = "image path error"
            rects = []
        else:
            rects = demo.detect(image)
            rects = rects.tolist()
            detect_msg = "success"
        end_time = time.time()
        results = json.dumps({"rects": rects, 
                              "cost_time": f"{end_time - start_time:.03f} s",
                              "detect_msg": detect_msg})
        conn.sendto(str.encode(results), addr)
    fdSocket.close()


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        '-p',
        type=int,
        default=8064,
        help='socket port',
    )
    return parser.parse_args()
    

def main():
    args = parser_args()
    print(args)
    port = args.port
    socket_start(port=port)


if __name__ == '__main__':
    main()
