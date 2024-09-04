import cv2
import numpy as np
import gradio as gr
from PIL import Image

from . import image_edit
from .image_edit import to_stego_image, extract_secret_image, edit_img, filter_process, apply_monochrome_filters, \
    apply_mosaic

# from .cutout_function import get_mask_image

# from .globals import history_mask, color_mask

history_mask = None
color_mask = None

# 交互式抠图实现函数
def get_mask_image(img):
    global color_mask
    ori_img = img["image"]
    ori_img = np.array(ori_img.convert('RGB'))  # 转换为 RGB 模式

    color_ranges = {
        'green': ((0, 255, 0), (0, 255, 0)),
        'red': ((255, 0, 0), (255, 0, 0)),
        'blue': ((0, 255, 255), (0, 255, 255)),
        'purple': ((255, 0, 255), (255, 0, 255))
    }

    # 存储各个类别遮罩的颜色
    grabcut_mask = np.full(color_mask.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)

    # 确保 color_mask 和 ori_img 大小一致
    if color_mask.shape[:2] != ori_img.shape[:2]:
        raise ValueError("color_mask 和 ori_img 的尺寸不匹配")

    # 分离每种颜色的遮罩
    for color, (lower, upper) in color_ranges.items():
        # 创建掩模
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(color_mask, lower_bound, upper_bound)
        # 根据颜色设置 GrabCut 的标签
        if color == 'green':
            grabcut_mask[mask == 255] = cv2.GC_FGD  # 确定的前景
        elif color == 'red':
            grabcut_mask[mask == 255] = cv2.GC_BGD  # 确定的背景
        elif color == 'blue':
            grabcut_mask[mask == 255] = cv2.GC_PR_FGD  # 可能的前景
        elif color == 'purple':
            grabcut_mask[mask == 255] = cv2.GC_PR_BGD  # 可能的背景

    # 初始化背景和前景模型
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # 使用 GrabCut 进行图像分割
    result_mask, bgdModel, fgdModel = cv2.grabCut(ori_img, grabcut_mask, None, bgdModel, fgdModel, 5,
                                                  cv2.GC_INIT_WITH_MASK)
    # result = ori_img * result_mask[:, :, np.newaxis]

    # 处理结果掩码
    result_mask = np.where((result_mask == cv2.GC_FGD) | (result_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    result = cv2.bitwise_and(ori_img, ori_img, mask=result_mask)

    image = Image.fromarray(result)
    # cv2.imshow('image', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result

# 动态创建文本框
def dummy_function(input_text):
    # 这个函数返回参数个数（可以是其他实际功能）
    return len(input_text.split())  # 例如，返回输入文本中单词的个数


# 处理图像选择事件的函数
# 暂时处理
def handle_image_select(image_index):
    print(image_index)


# 展示滤镜效果狂
# def filter_effects_display(img):
#     orig = Image.open('./ui/filter_effects_images/orig.jpg')  # 输出原图
#     images = [orig]
#
#     # # 创建按钮选择滤镜效果
#     # def create_button():
#     #     buttons = []
#     #     for index in range(len(images)):
#     #         button = gr.Button(value=f"Select Image {index + 1}")
#     #         buttons.append((button, index))
#     #     return buttons
#
#     with gr.Row():
#         # 创建滑动框
#         gallery = gr.Gallery(value=images,  label="Images", height=200, width=600,columns=3)
#         # buttons = create_button()
#         # for button, index in buttons:
#         #     button.click(fn=handle_image_select, inputs=[gr.State(value=index)], outputs=None)
#         #gallery.select(fn=handle_image_select,inputs )


# 根据输入的Json文件自动创建多个文本框以供微调
def create_textbox(num_boxes):
    textbox = []
    for i in range(num_boxes):
        with gr.Row() as row:
            # gr.Markdown(f"<center> 在这里修改第{i + 1}处的坐标:</center>")
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
                    # 原图
                    ori_image = gr.Image(label="Input Image", tool="color-sketch", visible=True)
                    # 获取图片颜色
                    get_color_image = gr.Image(label="Upload an image and click to pick a color", tool="editor",
                                               visible=False, interactive=True, scale=1)
                    # 马赛克
                    mosaic_image = gr.ImageMask(label="Mosaic", visible=False, interactive=True, scale=1, type='pil')
                    with gr.Row():
                        with gr.Tab("调节") as edit:
                            # 设置图像调节参数滑动条
                            with gr.Column():
                                brightness = gr.Slider(label="亮度", minimum=-50, maximum=50, step=0.01, value=0,
                                                       interactive=True)
                                Contrast = gr.Slider(label="对比度", minimum=0.3, maximum=3, step=0.01, value=1,
                                                     interactive=True)
                                Saturation = gr.Slider(label="饱和度", minimum=0.25, maximum=4, step=0.01, value=1,
                                                       interactive=True)
                                Sharpness = gr.Slider(label="锐度", minimum=0, maximum=10, step=0.01, value=0,
                                                      interactive=True)
                                Temperature = gr.Slider(label="色温", minimum=-30, maximum=30, step=0.1, value=0,
                                                        interactive=True)

                        with gr.Tab("滤镜"):
                            with gr.Row():
                                radio = gr.Radio(["原图", "锐利", "流年", "HDR", "反色", "美食", "冷艳", "单色"],
                                                 label="滤镜选择", info="请选择你感兴趣的滤镜")
                                # threshold = gr.Slider(minimum=0, maximum=100, step=0.01, value=10, visible=False)

                        with gr.Tab("马赛克") as mosaic:
                            with gr.Column():
                                weight = gr.Slider(minimum=0, maximum=30, step=1, value=10)
                                apply_button = gr.Button(value="应用效果")

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

                # 按下编辑按钮，更新交互界面
                def update_edit():
                    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

                # 按下单色按钮，更新交互界面
                def update_interface(filter_type):
                    if filter_type == '单色':
                        return gr.update(visible=False), gr.update(visible=True), gr.update(
                            visible=False)  # 原图框，单色图框，阈值滑动条框
                    else:
                        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

                # 按下马赛克按钮，更新界面
                def update_mosaic():
                    return gr.update(visible=False), gr.update(visible=True)

                # 监听参数调整
                setup_listeners(ori_image, brightness, Contrast, Saturation, Sharpness, Temperature, output_edit_image)

                # 界面更新
                edit.select(fn=update_edit, outputs=[ori_image, get_color_image, mosaic_image])
                radio.change(fn=update_interface, inputs=radio, outputs=[ori_image, get_color_image, mosaic_image])
                mosaic.select(fn=update_mosaic, outputs=[ori_image, mosaic_image])

                # 应用单色滤镜
                get_color_image.select(fn=apply_monochrome_filters, inputs=[get_color_image], outputs=output_edit_image)
                # 应用滤镜
                radio.change(fn=filter_process, inputs=[ori_image, radio], outputs=output_edit_image)

                apply_button.click(fn=apply_mosaic, inputs=[mosaic_image, weight], outputs=output_edit_image)

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
                    # 点击交互式抠图的按钮后，隐藏img_cutout，显示img_cutout_inter，可进行交互抠图。提醒：笔刷颜色需要是黑色/白色
                    ori_cutout_img = gr.Image(label="Output image", interactive=True, visible=True)
                    img_cutout_interactive = gr.Image(label="Background", source="upload", tool="sketch", type="pil",
                                                      height=512, brush_color='#42b983',
                                                      mask_opacity=0.5, brush_radius=100, visible=False,
                                                      interactive=True)

                    with gr.Row():
                        with gr.Column():
                            interactive_button = gr.Button(value="交互式抠图", visible=True)
                            exit_button = gr.Button(value="退出交互抠图模式", visible=False)
                        with gr.Column():
                            # 前景: (0, 255, 0)，背景:(255,0,0)，可能前景:(0, 255, 255)，可能背景:(255, 0, 255)
                            select_mask_mode = gr.Radio(['前景', '背景', '可能前景', '可能背景'], visible=False,
                                                        label="请选择mask属性", interactive=True)
                            with gr.Row():
                                ensure_mask = gr.Button(value="确定遮罩选择", visible=False, interactive=True)
                                ensure_button = gr.Button(value="确定", visible=False, interactive=True)
                        auto_cutout = gr.Button(value="自动抠图")
                        photo_make = gr.Button(value="证件照制作", visible=True)

                with gr.Column(scale=1):
                    cutout_color = gr.Image(label="Output image", interactive=False)
                    cutout_result = gr.Image(label="Output image", interactive=False, visible=False)
                    with gr.Row():
                        clear_mask_but = gr.Button(value="清除遮罩", visible=False, interactive=True)
                        gr.Button(value="Save")

                # 交互式抠图显示框,原抠图显示框 退出按钮,select_mask_mode,ensure_button,ensure_mask,自动抠图，证件照制作, cutout_result,clear_mask_but
                def update_cutout():
                    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(
                        visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
                        visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

                def recover_cutout():
                    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(
                        visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
                        visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

                # 前景: (0, 255, 0)，背景:(255,0,0)，可能前景:(0, 255, 255)，可能背景:(255, 0, 255)
                def get_cutout_color(radio):
                    brush_color = ['#00ff00', '#ff0000', '#00ffff', '#ff00ff']
                    if radio == "前景":
                        color = brush_color[0]
                    elif radio == "背景":
                        color = brush_color[1]
                    elif radio == "可能前景":
                        color = brush_color[2]
                    elif radio == "可能背景":
                        color = brush_color[3]
                    return color

                # 更新遮罩颜色
                def update_mask_mode(radio):
                    color = get_cutout_color(radio)
                    return gr.update(brush_color=color)

                # 确定遮罩颜色种类
                def get_mask(image, radio):
                    global history_mask, color_mask
                    color = get_cutout_color(radio)
                    ori_mask = image["mask"]
                    ori_mask = np.array(ori_mask)

                    if history_mask is None:
                        history_mask = np.zeros_like(ori_mask)
                    if color_mask is None:
                        color_mask = np.zeros_like(ori_mask)
                    # 十六进制转RGB
                    rgb_color = tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

                    # 接下来需要处理遮罩不同的部分，比较现有遮罩和历史遮罩的不同并重绘不同处
                    # 将新增的白色区域（假定为255, 255, 255）替换为指定颜色
                    white_mask = (ori_mask == 255).all(axis=-1)
                    # 找到新增的白色区域
                    new_white_mask = white_mask & ~(history_mask.all(axis=-1))
                    color_mask[new_white_mask] = rgb_color
                    # 更新历史遮罩
                    history_mask = ori_mask
                    # 将遮罩转换回图像格式
                    mask_image = Image.fromarray(color_mask)

                    return mask_image

                def clear_mask():
                    global history_mask, color_mask
                    tmp = np.zeros_like(color_mask)
                    history_mask = None
                    color_mask = None
                    return tmp

                interactive_button.click(fn=update_cutout,
                                         outputs=[ori_cutout_img, img_cutout_interactive, exit_button, select_mask_mode,
                                                  ensure_button, ensure_mask,
                                                  auto_cutout, photo_make, cutout_result, clear_mask_but])
                exit_button.click(fn=recover_cutout,
                                  outputs=[ori_cutout_img, img_cutout_interactive, exit_button, select_mask_mode,
                                           ensure_button, ensure_mask,
                                           auto_cutout, photo_make, cutout_result, clear_mask_but])
                select_mask_mode.change(fn=update_mask_mode, inputs=[select_mask_mode], outputs=img_cutout_interactive)

                ensure_mask.click(fn=get_mask, inputs=[img_cutout_interactive, select_mask_mode],
                                  outputs=[cutout_color])
                clear_mask_but.click(fn=clear_mask, outputs=[cutout_color])
                global history_mask, color_mask
                ensure_button.click(fn=get_mask_image, inputs=[img_cutout_interactive],
                                    outputs=[cutout_result])

        with gr.Tab("物品识别"):
            with gr.Row():
                with gr.Column():
                    gr.Image(label="输入待识别的图片", interactive=True)
                with gr.Column(scale=1):
                    gr.Image(label="识别结果", interactive=False)
                    with gr.Column():
                        with gr.Accordion("坐标微调", open=False):
                            textboxes = create_textbox(3)  # 效果展示

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
