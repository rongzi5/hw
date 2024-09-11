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


def get_face_key_point(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        face = faces[0]  # 只选择检测到的第一张脸
        landmarks = predictor(gray, face)
        # 获取左眼和右眼的中心点
        left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        left_eye_center = np.mean(left_eye_pts, axis=0).astype(int)
        right_eye_center = np.mean(right_eye_pts, axis=0).astype(int)
        return left_eye_center, right_eye_center
    else:
        return None, None


def bilinear_interpolation(img, vector_u, c):
    ux, uy = vector_u
    x1, x2 = int(ux), int(ux + 1)
    y1, y2 = int(uy), int(uy + 1)
    if x1 < 0 or x2 >= img.shape[1] or y1 < 0 or y2 >= img.shape[0]:
        return img[int(uy), int(ux), c]
    f_x_y1 = (x2 - ux) / (x2 - x1) * img[y1][x1][c] + (ux - x1) / (x2 - x1) * img[y1][x2][c]
    f_x_y2 = (x2 - ux) / (x2 - x1) * img[y2][x1][c] + (ux - x1) / (x2 - x1) * img[y2][x2][c]
    f_x_y = (y2 - uy) / (y2 - y1) * f_x_y1 + (uy - y1) / (y2 - y1) * f_x_y2
    return int(f_x_y)


def local_scaling_warps(img, cx, cy, r_max, a):
    img1 = np.copy(img)
    for y in range(cy - r_max, cy + r_max + 1):
        d = int(math.sqrt(r_max ** 2 - (y - cy) ** 2))
        x0 = cx - d
        x1 = cx + d
        for x in range(x0, x1 + 1):
            r = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            for c in range(3):
                vector_c = np.array([cx, cy])
                vector_r = np.array([x, y]) - vector_c
                f_s = (1 - ((r / r_max - 1) ** 2) * a)
                vector_u = vector_c + f_s * vector_r
                img1[y][x][c] = bilinear_interpolation(img, vector_u, c)
    return img1


def big_eye(img, r_max, a, left_eye_pos=None, right_eye_pos=None):
    img0 = img.copy()
    if left_eye_pos is None or right_eye_pos is None:
        left_eye_pos, right_eye_pos = get_face_key_point(img0)
    if left_eye_pos is not None and right_eye_pos is not None:
        img0 = local_scaling_warps(img0, left_eye_pos[0], left_eye_pos[1], r_max, a)
        img0 = local_scaling_warps(img0, right_eye_pos[0], right_eye_pos[1], r_max, a)
    return img0


# 示例使用函数
def update_image_with_slider_bigeyes(image, slider_value):
    a = slider_value / 150  # 根据滑动条的值计算放大倍率
    modified_image = big_eye(image, 60, a)  # 调用big_eye函数处理图像
    return modified_image
