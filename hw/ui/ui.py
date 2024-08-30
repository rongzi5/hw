import cv2
import numpy as np
import gradio as gr

from . import function
from .function import to_stego_image, extract_secret_image, edit_img


#动态创建文本框
def dummy_function(input_text):
    # 这个函数返回参数个数（可以是其他实际功能）
    return len(input_text.split())  # 例如，返回输入文本中单词的个数


#根据输入的Json文件自动创建多个文本框以供微调
def create_textbox(num_boxes):
    textbox = []
    for i in range(num_boxes):
        with gr.Row() as row:
            #gr.Markdown(f"<center> 在这里修改第{i + 1}处的坐标:</center>")
            gr.Textbox(scale=1, label=f"{i + 1}处的坐标")
        textbox.append(row)
    return textbox


def create_ui():
    with gr.Blocks() as ui:
        # 图像基础编辑
        with gr.Tab("图像编辑"):
            with gr.Row():
                # 左边列：固定宽度
                with gr.Column():
                    ori_image = gr.Image(label="Input Image", tool="color-sketch")
                    with gr.Row():
                        with gr.Tab("调节"):
                            # 设置图像调节参数滑动条
                            with gr.Column():
                                brightness = gr.Slider(label="亮度", minimum=-50, maximum=50, step=0.01, value=0, interactive=True)
                                Contrast = gr.Slider(label="对比度", minimum=0.3, maximum=3, step=0.01, value=1, interactive=True)
                                Saturation = gr.Slider(label="饱和度", minimum=0.25, maximum=4, step=0.01, value=1, interactive=True)
                                Sharpness = gr.Slider(label="锐度", minimum=0, maximum=10, step=0.01, value=0, interactive=True)
                                Temperature = gr.Slider(label="色温", minimum=-30, maximum=30, step=0.1, value=0, interactive=True)

                        with gr.Tab("滤镜"):
                            pass
                        with gr.Tab("马赛克"):
                            gr.ImageMask()

                # 右边列：自适应
                with gr.Column(scale=1):
                    with gr.Column():
                        output_edit_image = gr.Image(label="Output image", interactive=False)
                        gr.Button(value="Save")

                def setup_listeners(image, bri, contrast, saturation, sharpness, temperature, result):
                    # 将所有控件的变化事件绑定到 update_image 函数
                    controls = [image, bri, contrast, saturation, sharpness, temperature]
                    for control in controls:
                        control.change(
                            fn=edit_img,
                            inputs=[image, brightness, contrast, saturation, sharpness, temperature],
                            outputs=result,
                        )

                setup_listeners(ori_image, brightness, Contrast, Saturation, Sharpness, Temperature,output_edit_image)

        with gr.Tab("美颜"):
            with gr.Row():
                with gr.Column():
                    gr.Image()
                    with gr.Tab("参数调节"):
                        gr.Slider(label="大眼", minimum=0, maximum=100, step=0.1, value=0.1, interactive=True)
                        gr.Slider(label="瘦脸", minimum=0, maximum=100, step=0.1, value=0.1, interactive=True)
                        gr.Slider(label="磨皮", minimum=0, maximum=100, step=0.1, value=0.1, interactive=True)
                with gr.Column(scale=1):
                    gr.Image()

        #抠图设置
        with gr.Tab("抠图"):
            with gr.Row():
                with gr.Column():
                    #
                    #点击交互式抠图的按钮后，隐藏img_cutout，显示img_cutout_inter，可进行交互抠图。提醒：笔刷颜色需要是黑色/白色
                    img_cutout_inter = gr.Image(label="Input image", interactive=True, visible=False, tool="editor")
                    img_cutout = gr.Image(label="Output image", interactive=True, visible=True)
                    with gr.Row():
                        gr.Button(value="自动抠图")
                        gr.Button(value="交互式抠图")
                        gr.Button(value="证件照制作")

                with gr.Column(scale=1):
                    gr.Image(label="Output image", interactive=False)
                    gr.Button(value="Save")
        with gr.Tab("物品识别"):
            with gr.Row():
                with gr.Column():
                    gr.Image(label="输入待识别的图片", interactive=True)
                with gr.Column(scale=1):
                    gr.Image(label="识别结果", interactive=False)
                    with gr.Column():
                        with gr.Accordion("坐标微调", open=False):
                            textboxes = create_textbox(3)  #效果展示

        with gr.Tab("发现"):
            with gr.Tab("风格迁移"):
                with gr.Row():
                    with gr.Column():
                        gr.Image()
                    with gr.Column(scale=1):
                        gr.Image()

            with gr.Tab("图片隐写还原"):
                with gr.Row():
                    with gr.Column():
                        carrier_img = gr.Image()
                        secret_img = gr.Image()
                        with gr.Row():
                            stego_button = gr.Button(value="隐写")
                            extract_button = gr.Button(value="还原")
                    with gr.Column(scale=1):
                        process_image = gr.Image(label="隐写图像输出/输入", interactive=True)
                        save_button = gr.Button(value="保存")

            stego_button.click(fn=to_stego_image, inputs=[carrier_img, secret_img], outputs=process_image)
            extract_button.click(fn=extract_secret_image, inputs=process_image, outputs=[carrier_img, secret_img])
            with gr.Tab("文字识别"):
                with gr.Row():
                    with gr.Column():
                        gr.Image()
                    with gr.Column(scale=1):
                        gr.Textbox()
            with gr.Tab("自动去水印"):
                with gr.Row():
                    with gr.Column():
                        gr.Image()
                    with gr.Column(scale=1):
                        gr.Image()
    return ui
