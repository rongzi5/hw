import cv2
import gradio as gr
import os


# 打开存储文件夹路径
def open_file_folder(path: str):
    print(f"Open {path}")
    if path is None or path == "":
        return

    command = f'explorer /select,"{path}"'
    os.system(command)


# 处理灰度图
def gray_to_rgb(gray_img):
    return cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)


# 调整载体图像的尺寸以匹配秘密图像的尺寸，并返回调整后的载体图像。
def img_resize(img1, img2):
    flag = 0
    if img1.shape == img2.shape:
        return flag, img1, img2
    else:
        flag = 1
        if img1.shape[0] * img1.shape[1] > img2.shape[0] * img2.shape[1]:
            img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
        else:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        return flag, img1, img2


# 实现图片取色
def get_color(image, x, y):
    color = image[y, x]
    # color_rgb = color[::-1]  # BGR 转换为 RGB
    # color_hex = '#{:02x}{:02x}{:02x}'.format(color_rgb[0], color_rgb[1], color_rgb[2])
    r, g, b = color[0], color[1], color[2]
    return r, g, b


# 返回点击位置图片的坐标并返回色号
def get_image_color(image, evt: gr.SelectData):
    x, y = int(evt.index[0]), int(evt.index[1])
    # print(x, y)
    return get_color(image, x, y)
