import cv2
import numpy as np


def smoothing(img, slider_value):
    # 根据滑动条值计算sigmaColor和sigmaSpace
    sigmaColor = slider_value / 2 * 3
    sigmaSpace = slider_value / 2 * 3

    if sigmaColor == 0 and sigmaSpace == 0:
        return img  # 如果滑动条值为0，直接返回原图
    # 应用双边滤波
    new_image = cv2.bilateralFilter(img, 12, sigmaColor, sigmaSpace)
    return new_image


def process_image_with_slider_smooth(image, slider_value):
    # 使用滑动条值处理图像
    output = smoothing(image, slider_value)
    return output
