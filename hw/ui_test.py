import gradio as gr
import cv2
import numpy as np
from PIL import Image


# 对比度调整函数
def gama_transfer(img, power1):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = 255 * np.power(img / 255, power1)
    img = np.around(img)
    img[img > 255] = 255
    out_img = img.astype(np.uint8)
    return out_img


# 图像处理函数，用于 Gradio 接口
def process_image(image, contrast):
    # 将 PIL 图像转换为 NumPy 数组
    image_np = np.array(image)
    # 调整对比度
    result_img = gama_transfer(image_np, contrast)
    # 将结果转换回 PIL 图像
    return Image.fromarray(result_img)


# 创建 Gradio 界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            # 上传图像
            image = gr.Image(label="Input Image", type="pil")
            # 滑动条调整对比度
            gama_transfer_num = gr.Slider(label="对比度", minimum=-10, maximum=10, step=0.1, value=0.1,
                                          interactive=True)
        with gr.Column():
            # 输出调整后的图像
            output = gr.Image(label="Output Image")


    # 实时处理图像
    def update_image(image, contrast):
        return process_image(image, contrast)


    # 监听图像和滑动条的变化
    image.change(
        fn=update_image,
        inputs=[image, gama_transfer_num],
        outputs=output,
    )
    gama_transfer_num.change(
        fn=update_image,
        inputs=[image, gama_transfer_num],
        outputs=output,
    )

demo.launch()
