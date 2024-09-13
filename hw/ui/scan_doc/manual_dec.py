import cv2
import gradio as gr
import numpy as np


def manual_dec(image, index, xy_position, evt: gr.SelectData):
    # 确定点击的位置并更新 xy_position 数组
    x, y = int(evt.index[0]), int(evt.index[1])
    xy_position[index] = [x, y]

    # 当 index >= 1 时，绘制从第一个点到当前点的线
    if index >= 1:
        for i in range(1, index + 1):
            # 直接计算整数元组，减少 map(int, ...) 的重复调用
            pt1 = tuple(map(int, xy_position[i]))
            pt2 = tuple(map(int, xy_position[i - 1]))
            image = cv2.line(image, pt1, pt2, (0, 255, 0), 20)

            # 当绘制完第4个点时，闭合图形
            if i == 3:
                pt3 = tuple(map(int, xy_position[0]))
                image = cv2.line(image, pt1, pt3, (0, 255, 0), 20)

    index += 1
    return image, xy_position, index


def four_point_transform(img, xy_position):
    # 将 xy_position 转为 float32 并直接使用 np.linalg.norm 优化距离计算
    xy_position = np.array(xy_position, dtype="float32")

    # 计算宽度和高度
    widthA = np.linalg.norm(xy_position[1] - xy_position[0])
    widthB = np.linalg.norm(xy_position[2] - xy_position[3])
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(xy_position[0] - xy_position[3])
    heightB = np.linalg.norm(xy_position[1] - xy_position[2])
    maxHeight = max(int(heightA), int(heightB))

    # 定义目标点位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32"
    )

    # 获取透视变换矩阵并应用变换
    M = cv2.getPerspectiveTransform(xy_position, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # 转换为灰度图并使用自适应阈值进行二值化
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    result = cv2.adaptiveThreshold(gray_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return result
