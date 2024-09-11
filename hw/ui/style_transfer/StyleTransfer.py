import cv2 as cv
import numpy as np
import os


def style_transfer(image, model_name):
    # 文件位置
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 模型位置
    model_path = os.path.join(script_dir, 'models', f'{model_name}.t7')

    # 检查模型文件是否存在
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    # 加载模型
    net = cv.dnn.readNetFromTorch(model_path)

    # 读取图像
    frame = image

    # 获取图像尺寸
    (h, w) = frame.shape[:2]

    # 设定输入尺寸
    inWidth = 600
    inHeight = int((inWidth / w) * h)

    # 处理图像
    inp = cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),
                               (103.939, 116.779, 123.68), swapRB=False, crop=False)

    # 风格迁移
    net.setInput(inp)
    out = net.forward()

    # 模型输出
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0] += 103.939
    out[1] += 116.779
    out[2] += 123.68
    out /= 255
    out = out.transpose(1, 2, 0)

    # 返回处理后的图像
    return (out * 255).astype(np.uint8)

# if __name__ == "__main__":
#     try:
#         result_image = style_transfer('input_image.jpg', 'candy')
#         cv.imshow('Styled Image', result_image)
#         cv.waitKey(0)
#         cv.destroyAllWindows()
#     except Exception as e:
#         print(f"An error occurred: {e}")
