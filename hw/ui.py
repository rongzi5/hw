import numpy as np
import gradio as gr

with gr.Blocks() as ui:
    # 图像基础编辑
    with gr.Tab("图像编辑"):
        with gr.Row():
            # 左边列：固定宽度
            with gr.Column():
                image = gr.ImageEditor(label="Input Image", sources=["upload"])
                with gr.Row():
                    with gr.Tab("调节"):
                        #设置图像调节参数滑动条
                        with gr.Column():
                            gr.Slider(label="对比度", minimum=0, maximum=100, step=0.1, value=0.1, interactive=True)
                            gr.Slider(label="饱和度", minimum=0, maximum=100, step=0.1, value=0.1, interactive=True)
                            gr.Slider(label="锐度", minimum=0, maximum=100, step=0.1, value=0.1, interactive=True)
                            gr.Slider(label="亮度", minimum=0, maximum=100, step=0.1, value=0.1, interactive=True)
                            gr.Slider(label="色温", minimum=0, maximum=100, step=0.1, value=0.1, interactive=True)
                    with gr.Tab("滤镜"):
                        pass
                    with gr.Tab("马赛克"):
                        gr.ImageMask()

            # 右边列：自适应
            with gr.Column(scale=1):
                with gr.Column():
                    output = gr.Image(label="Output image", interactive=False)
                    gr.Button(value="Save")
    #抠图设置
    #无法更换颜色？？？考虑弹出opencv窗口进行操作
    with gr.Tab("抠图"):
        with gr.Row():
            with gr.Column():
                #
                #点击交互式抠图的按钮后，隐藏img_cutout，显示img_cutout_inter，可进行交互抠图
                img_cutout_inter = gr.ImageMask(
                    label="Input image",
                    brush=gr.Brush(colors=["#000000", "#FFFFFF"],   color_mode="fixed"),
                    interactive=True,visible=False
                )
                img_cutout = gr.Image(label="Output image", interactive=True,visible=True)
                with gr.Row():
                    gr.Button(value="自动抠图")
                    gr.Button(value="交互式抠图")
                    gr.Button(value="证件照制作")

            with gr.Column(scale=1):
                gr.Image(label="Output image", interactive=False)
                gr.Button(value="Save")
    with gr.Tab("物品识别"):
        with gr.Row():
            gr.Image(label="输入待识别的图片", interactive=True)
            gr.Image(label="识别结果",interactive=False)

ui.launch()
