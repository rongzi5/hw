import cv2
import numpy as np


def edge_preserving_smoothing(image, sigma_s, sigma_r):
    """
    接受图像和滑动条值，对图像进行边缘保持滤波处理。

    参数:
    image: 输入图像 (BGR格式)
    sigma_s: 固定的sigma_s参数
    sigma_r: 滑动条值，对应边缘保持滤波器中的sigma_r参数，范围应为0到1

    返回值:
    处理后的图像
    """
    # 使用边缘保持滤波器进行平滑处理
    smoothed_image = cv2.edgePreservingFilter(image, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
    return smoothed_image


def process_image_with_fixed_sigma_s(image, slider_value):
    """
    使用固定的 sigma_s 值和可调的 sigma_r 值处理图像。

    参数:
    image: 输入图像
    slider_value: 滑动条值 (用于计算sigma_r)

    返回值:
    处理后的图像
    """
    sigma_s = 50  # 固定 sigma_s 为 50
    sigma_r = slider_value / 100.0  # sigma_r 范围在 0 到 1 之间
    processed_image = edge_preserving_smoothing(image, sigma_s, sigma_r)
    return processed_image
