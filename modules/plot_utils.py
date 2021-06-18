import os
import cv2
import random
import numpy as np
from PIL import Image, ImageFont, ImageDraw
working_root = os.path.split(os.path.realpath(__file__))[0]


def draw_text(frame, point, text, color):
    font = ImageFont.truetype(os.path.join(working_root, 'simsun.ttc'), 64)
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.text(point, text, fill=color, font=font)
    frame[::] = np.array(image)


def draw_detection_rects(image: np.ndarray, detection_rects: np.ndarray, colors=None, names=None, method=1):
    if not isinstance(detection_rects, np.ndarray):
        detection_rects = np.array(detection_rects)
    if method:
        width = image.shape[1]
        height = image.shape[0]
    else:
        width = 1.0
        height = 1.0
    if colors is None:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(10)]
    for index in range(detection_rects.shape[0]):
        if detection_rects.shape[1] == 6:
            if len(colors) <= int(detection_rects[index, 5]):
                colors.extend([[random.randint(0, 255) for _ in range(3)] for _ in range(int(detection_rects[index, 5]) - len(colors) + 2)])
            color = colors[int(detection_rects[index, 5])]
            thickness = 2
        else:
            color = colors[0]
            thickness = 2
        cv2.rectangle(image,
                      (int(detection_rects[index, 0] * width), int(detection_rects[index, 1] * height)),
                      (int(detection_rects[index, 2] * width), int(detection_rects[index, 3] * height)),
                      color,
                      thickness=thickness)
        if detection_rects.shape[1] >= 5:
            if detection_rects.shape[1] == 6:
                label_id = int(detection_rects[index, 5])
                if names is not None and label_id < len(names):
                    msg = f"{names[label_id]}: {detection_rects[index, 4]:.03f}"
                else:
                    msg = f"{label_id}: {detection_rects[index, 4]:.03f}"
            else:
                msg = f"{detection_rects[index, 4]:.03f}"
            cv2.putText(image, msg,
                        (int(detection_rects[index, 0] * width), int(detection_rects[index, 1] * height)),
                        1, 1, color)
    return image


def draw_landmarks(image, landmarks, norm=True):
    """

    Parameters
    ----------
    image 展示的原始图片
    landmarks 维度为[n,106, 2]的列表或者numpy数组
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
    for num in range(landmarks.shape[0]):
        landmark = landmarks[num]
        for index in range(landmark.shape[0]):
            pt1 = (int(scale_width * landmark[index, 0]), int(scale_height * landmark[index, 1]))
            cv2.circle(image, pt1, 1, (0, 0, 255), 2)
        plot_line = lambda i1, i2: cv2.line(image,
                                            (int(scale_width * landmark[i1, 0]),
                                            int(scale_height * landmark[i1, 1])),
                                            (int(scale_width * landmark[i2, 0]),
                                            int(scale_height * landmark[i2, 1])),
                                            (255, 255, 255), 1)
        close_point_list = [0, 33, 42, 51, 55, 66, 74, 76, 84, 86, 98, 106]
        for ind in range(len(close_point_list) - 1):
            l, r = close_point_list[ind], close_point_list[ind + 1]
            for index in range(l, r - 1):
                plot_line(index, index + 1)
            # 将眼部, 嘴部连线闭合
            plot_line(41, 33)  # 左眉毛
            plot_line(50, 42)  # 右眉毛
            plot_line(65, 55)  # 鼻子
            plot_line(73, 66)  # 左眼
            plot_line(83, 76)  # 右眼
            plot_line(97, 86)  # 外唇
            plot_line(105, 98)  # 内唇


def draw_segmentation_mask(image: np.ndarray, masks: np.ndarray, classes=None, colors=None):
    if colors is None:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(10)]
    if len(colors) <= int(masks.shape[0]):
        colors.extend([[random.randint(0, 255) for _ in range(3)] for _ in range(masks.shape[0] - len(colors) + 2)])
    colors_map = {}
    for index in range(masks.shape[0]):
        class_id = index if classes is None else classes[index]
        if class_id not in colors_map:
            colors_map.update({class_id: colors[index]})
    # print(colors_map)
    image = image.astype(float)
    for index in range(masks.shape[0]):
        class_id = index if classes is None else classes[index]
        mask = masks[index]
        mask[mask>0.1] = 1.0
        mask[mask<0.1] = 0
        alpha = np.stack([mask, mask, mask], 2)
        alpha = cv2.resize(alpha, (image.shape[1], image.shape[0]))
        alpha = alpha.astype(float)
        foreground = cv2.multiply(alpha, image)
        background = cv2.multiply(1.0 - alpha, image)
        # background = image - foreground
        color_foreground = cv2.addWeighted(foreground, 0.5, colors_map[class_id] * alpha, 0.5, 0)
        image = cv2.add(background, color_foreground)
        # cv2.imshow("background", background)
        # cv2.imshow("foreground", foreground)
        # cv2.imshow("color_foreground", color_foreground)
        # cv2.waitKey()
    image = image.astype(np.uint8, copy=True)
    return image


def crop_segmentation_mask(foreground: np.ndarray, background: np.ndarray, mask: np.ndarray, box: np.ndarray, scale=1.0, angle=0, x=0, y=0):
    height, width = background.shape[0:2]
    assert (x >= 0 and  y >= 0 and  x < width and y < height), "x, y value is invalid !"
    foreground = foreground.astype(float)
    background = background.astype(float)
   
    mask[mask>0.1] = 1.0
    mask[mask<0.1] = 0
    alpha = np.stack([mask, mask, mask], 2)
    alpha = cv2.resize(alpha, (foreground.shape[1], foreground.shape[0]))
    alpha = alpha.astype(float)
    foreground = cv2.multiply(alpha, foreground)

    box_w = int((box[2] - box[0]) * foreground.shape[1])
    box_h = int((box[3] - box[1]) * foreground.shape[0])
    top_left_x =  int(box[0] * foreground.shape[1])
    top_left_y =  int(box[1] * foreground.shape[0])

    # 裁剪包含掩码部分的目标框图像
    crop_alpha = alpha[top_left_y:top_left_y+box_h, top_left_x: top_left_x+box_w]
    crop_foreground = foreground[top_left_y:top_left_y+box_h, top_left_x: top_left_x+box_w]
    # 掩码框旋转
    center = (int(box_w /2), int(box_h /2))
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数）,建议0.75
    rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    cos = np.abs(rotate_matrix[0, 0])
    sin = np.abs(rotate_matrix[0, 1])
 
    # compute the new bounding dimensions of the image
    rotate_w = int((box_h * sin) + (box_w * cos))
    rotate_h = int((box_h * cos) + (box_w * sin))
 
    # adjust the rotation matrix to take into account translation
    rotate_matrix[0, 2] += (rotate_w / 2) - center[0]
    rotate_matrix[1, 2] += (rotate_h / 2) - center[1]

    rotate_alpha = cv2.warpAffine(crop_alpha, rotate_matrix, (rotate_w, rotate_h), borderValue=(0, 0, 0))
    rotate_foreground = cv2.warpAffine(crop_foreground, rotate_matrix, (rotate_w, rotate_h) , borderValue=(0, 0, 0))
    # 掩码框缩放
    scale_w = int(rotate_w * scale)
    scale_h = int(rotate_h * scale)
    scale_alpha = cv2.resize(rotate_alpha, (scale_w, scale_h))
    scale_foreground = cv2.resize(rotate_foreground, (scale_w, scale_h))
    scale_alpha = scale_alpha[0: min(height - y, scale_h), 0: min(width - x, scale_w)]
    scale_foreground = scale_foreground[0: min(height - y, scale_h), 0: min(width - x, scale_w)]
    # cv2.imshow("scale_foreground", scale_foreground/255.0)
    # cv2.waitKey()
    
    scale_alpha = scale_alpha.astype(float)
    crop_background = background[y: y+scale_h, x: x+scale_w].copy()
    crop_background = cv2.multiply(1.0 - scale_alpha, crop_background)
    background[y: y+scale_h, x: x+scale_w] = cv2.add(crop_background, scale_foreground)
    # cv2.imshow("background", crop_background)
    # cv2.imshow("foreground", scale_foreground/255)
    # cv2.imshow("color_foreground", color_foreground)
    # cv2.waitKey()
    background = background.astype(np.uint8, copy=True)
    return background
