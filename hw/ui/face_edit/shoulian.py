import cv2
import dlib
import numpy as np
import math
import os

# 初始化人脸检测器和特征点预测器
detector = dlib.get_frontal_face_detector()
# 获取当前文件所在的目录
current_dir = os.path.dirname(__file__)
# 构造shape_predictor文件的绝对路径
path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(path)


def bilinear_interpolation(img, vector_u, c):
    ux, uy = vector_u
    x1, x2 = int(ux), int(ux + 1)
    y1, y2 = int(uy), int(uy + 1)
    if x1 < 0 or x2 >= img.shape[1] or y1 < 0 or y2 >= img.shape[0]:
        return img[int(uy), int(ux), c]
    f_x_y1 = (x2 - ux) * img[y1, x1, c] + (ux - x1) * img[y1, x2, c]
    f_x_y2 = (x2 - ux) * img[y2, x1, c] + (ux - x1) * img[y2, x2, c]
    f_x_y = (y2 - uy) * f_x_y1 + (uy - y1) * f_x_y2
    return int(f_x_y)


def localTranslationWarp(srcImg, startX, startY, endX, endY, radius):
    ddradius = float(radius * radius)
    copyImg = srcImg.copy()
    ddmc = (endX - startX) ** 2 + (endY - startY) ** 2
    H, W, C = srcImg.shape
    for i in range(W):
        for j in range(H):
            if (i - startX) ** 2 + (j - startY) ** 2 > radius ** 2:
                continue
            distance = (i - startX) ** 2 + (j - startY) ** 2
            if distance < ddradius:
                ratio = (ddradius - distance) / (ddradius - distance + ddmc)
                ratio = ratio ** 2
                UX = i - ratio * (endX - startX)
                UY = j - ratio * (endY - startY)
                UX = max(0, min(W - 1, UX))
                UY = max(0, min(H - 1, UY))
                for c in range(C):
                    copyImg[j, i, c] = bilinear_interpolation(srcImg, (UX, UY), c)
    return copyImg


def face_thin_auto(src, adjustment=1.0):
    landmarks = landmark_dec_dlib_fun(src)
    if len(landmarks) == 0:
        return src  # 如果没有检测到人脸则返回原图
    thin_image = src.copy()
    for landmarks_node in landmarks:
        left_landmark = landmarks_node[3]
        left_landmark_down = landmarks_node[5]
        right_landmark = landmarks_node[13]
        right_landmark_down = landmarks_node[15]
        endPt = landmarks_node[30]

        r_left = np.linalg.norm(left_landmark - left_landmark_down)
        r_right = np.linalg.norm(right_landmark - right_landmark_down)

        # 应用用户指定的调节参数，基于自动计算的参数进行调整
        r_left_adjusted = r_left * adjustment
        r_right_adjusted = r_right * adjustment

        thin_image = localTranslationWarp(thin_image, left_landmark[0, 0], left_landmark[0, 1], endPt[0, 0],
                                          endPt[0, 1], r_left_adjusted)
        thin_image = localTranslationWarp(thin_image, right_landmark[0, 0], right_landmark[0, 1], endPt[0, 0],
                                          endPt[0, 1], r_right_adjusted)
    return thin_image


def landmark_dec_dlib_fun(img_src):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    rects = detector(img_gray, 0)
    land_marks = []
    for rect in rects:
        land_marks_node = np.matrix([[p.x, p.y] for p in predictor(img_gray, rect).parts()])
        land_marks.append(land_marks_node)
    return land_marks
