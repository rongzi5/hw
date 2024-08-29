import cv2
import gradio as gr
import os

#打开存储文件夹路径
def open_file_folder(path: str):
    print(f"Open {path}")
    if path is None or path == "":
        return

    command = f'explorer /select,"{path}"'
    os.system(command)

#处理灰度图
def gray_to_rgb(gray_img):
    return cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)


#调整载体图像的尺寸以匹配秘密图像的尺寸，并返回调整后的载体图像。
def img_resize(img1,img2):
    flag = 1
    if img1.shape == img2.shape:
        return flag, img1, img2
    else:
        flag = 0
        if img1.shape[0]*img1.shape[1] > img2.shape[0]*img2.shape[1]:
            img1 = cv2.resize(img1,(img2.shape[1],img2.shape[0]))
        else:
            img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]))
        return flag,img1,img2

