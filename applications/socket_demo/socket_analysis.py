from socket import *
import os
import json
import glob
import argparse


def get_demo_socket(image_path=None, host='127.0.0.1', port=8064):
    client_socket = socket(AF_INET, SOCK_STREAM)
    image_path = os.path.abspath(image_path)
    try:
        client_socket.connect((host, port))
        client_socket.send(str.encode(image_path))
        data = client_socket.recv(1024)
        results = data.decode('utf-8')
        results = json.loads(results)
    except Exception:
        results = {}
    client_socket.close()
    return results


def get_folder_demo_socket(image_folder_path=None, host='127.0.0.1', port=8064):
    for image_path in sorted(glob.glob(image_folder_path + "/*.jpg")):
        print(image_path)
        results = get_demo_socket(image_path=image_path, host=host, port=port)
        print(results)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        '-p',
        type=int,
        default=8064,
        help='socket port',
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='socket host',
    )
    parser.add_argument(
        '--image_path',
        '-i',
        type=str,
        default=None,
        help='image path to inference',
    )
    return parser.parse_args()


def main():
    args = parser_args()
    print(args)
    if not os.path.isdir(args.image_path):
        results = get_demo_socket(image_path=args.image_path, host=args.host, port=args.port)
        print(results)
    else:
        get_folder_demo_socket(image_folder_path=args.image_path, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
