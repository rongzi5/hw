import cv2
import numpy as np
import dlib
import os

# 初始化人脸检测器和特征点预测器
detector = dlib.get_frontal_face_detector()
# path = "D:/Users/15654/PycharmProjects/hw/hw/ui/face_edit/shape_predictor_68_face_landmarks.dat"
# 获取当前文件所在的目录
current_dir = os.path.dirname(__file__)
# 构造shape_predictor文件的绝对路径
path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(path)


def guided_filter(I, p, win_size, eps):
    mean_I = cv2.blur(I, (win_size, win_size))
    mean_p = cv2.blur(p, (win_size, win_size))

    corr_I = cv2.blur(I * I, (win_size, win_size))
    corr_Ip = cv2.blur(I * p, (win_size, win_size))

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.blur(a, (win_size, win_size))
    mean_b = cv2.blur(b, (win_size, win_size))

    q = mean_a * I + mean_b

    return q


def YCrCb_ellipse_model(img):
    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(skinCrCbHist, (113, 155), (23, 25), 43, 0, 360, (255, 255, 255), -1)

    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (Y, Cr, Cb) = cv2.split(YCrCb)
    skin = np.zeros(Cr.shape, dtype=np.uint8)

    for i in range(Cr.shape[0]):
        for j in range(Cr.shape[1]):
            if skinCrCbHist[Cr[i, j], Cb[i, j]] > 0:
                skin[i, j] = 255

    res = cv2.bitwise_and(img, img, mask=skin)
    return skin, res


def strengthen_light(img, light):
    skin, _ = YCrCb_ellipse_model(img)
    kernel = np.ones((3, 3), dtype=np.uint8)
    skin = cv2.erode(skin, kernel=kernel)
    skin = cv2.dilate(skin, kernel=kernel)

    img1 = guided_filter(img / 255.0, img / 255.0, 10, 0.001) * 255
    img1 = np.array(img1, dtype=np.uint8)
    img1 = cv2.bitwise_and(img1, img1, mask=skin)
    skin = cv2.bitwise_not(skin)
    img1 = cv2.add(img1, cv2.bitwise_and(img, img, mask=skin))

    h, w = img.shape[:2]
    for i in range(0, h):
        for j in range(0, w):
            b, g, r = img[i, j]
            img1[i, j] = (min(b + light, 255), min(g + light, 255), min(r + light, 255))
    return img1


# 封装的函数，使用图像和滑动条值
def process_image_with_slider_white(image, slider_value):
    output = strengthen_light(image, slider_value)
    return output
