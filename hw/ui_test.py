import cv2
import numpy as np
from PIL import Image
import gradio as gr


def apply_opencv_effect(image):
    # 将 PIL 图像转换为 OpenCV 格式
    image = np.array(image)

    # 在 OpenCV 中应用窗口效果，比如这里的高斯模糊
    processed_image = cv2.GaussianBlur(image, (15, 15), 0)

    # 将 OpenCV 图像转换回 PIL 格式
    processed_image = Image.fromarray(processed_image)

    return processed_image


# 创建 Gradio 接口
with gr.Blocks() as demo:
    with gr.Row():
        input_image = gr.Image(label="Upload Image", type="pil")
        output_image = gr.Image(label="Processed Image")

    # 连接组件
    input_image.change(fn=apply_opencv_effect, inputs=input_image, outputs=output_image)

# 启动 Gradio 应用
demo.launch()
