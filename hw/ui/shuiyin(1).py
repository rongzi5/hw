import cv2
import numpy as np

# 读取模板图像
image = cv2.imread("test2.jpg")

w0,h0,c0=image.shape

roi = cv2.selectROI(image, showCrosshair=True, fromCenter=False)
x, y, w, h = roi
rect=(x,y,w,h)
roi_image = image[int(y):int(y+h), int(x):int(x+w)]

# hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
# lower_white = np.array([0, 0, 0], dtype=np.uint8)
# upper_white = np.array([255, 120, 255], dtype=np.uint8)
# roi_image1 = cv2.inRange(hsv, lower_white, upper_white)
# roi_image[roi_image1 > 0] = [220,220,220]
# roi_image[roi_image1 == 0] = [255, 255, 255]
# cv2.imshow('roiu', roi_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


src=cv2.imread("test2.jpg")
mask = np.zeros((w0, h0, c0), dtype=np.uint8) * 255
mask[int(y):int(y+h), int(x):int(x+w)] = roi_image
mask=mask[:,:,0]
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('scr', src)
cv2.waitKey(0)
cv2.destroyAllWindows()

def Remove_watermark(image,mask):
    hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_range = np.array([140, 100, 90])
    high_range = np.array([185, 255, 255])
    #mask = cv2.inRange(hue_image, low_range, high_range)

    kernel = np.ones((3, 3), np.uint8)
    dilate_img = cv2.dilate(mask, kernel, iterations=1)
    res = cv2.inpaint(image,dilate_img,5,flags=cv2.INPAINT_TELEA)

    cv2.imshow('mask_img',mask)
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

Remove_watermark(src,mask)

# src = cv2.resize(src,None,fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
# mask = cv2.resize(mask,None,fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
# save = np.zeros(src.shape, np.uint8) #创建一张空图像用于保存
# for row in range(src.shape[0]):
#     for col in range(src.shape[1]):
#         for channel in range(src.shape[2]):
#             if mask[row, col, channel] == 0:
#                 val = 0
#             else:
#                 reverse_val = 255 - src[row, col, channel]
#                 val = 255 - reverse_val * 256 / mask[row, col, channel]
#                 if val < 0: val = 0
#             save[row, col, channel] = val



# kernel_size = (5, 5)
# sigma = 1.0
# kernel = np.ones((5, 5), np.float32) /25
# smoothed_image = cv2.filter2D(save, -1, kernel)
# smoothed_image = cv2.filter2D(smoothed_image, -1, kernel)
# blank_image = np.ones(smoothed_image.shape, dtype=np.uint8) * 127
# smoothed_image = cv2.addWeighted(smoothed_image, 1, blank_image, 0.5, 0)
#save[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = smoothed_image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]


# cv2.imshow('src', src)
# cv2.imshow('mask', mask)
# cv2.imshow('save', save)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

