import cv2
import numpy as np
import gradio as gr
import math
from PIL import Image

from .image_edit import to_stego_image, extract_secret_image, edit_img, filter_process, apply_monochrome_filters, \
    apply_mosaic
from .getface import get_prediction
from .utils import draw_process
from .shuiyin import Remove_watermark

from .face_edit import beauty_image_processing
from .style_transfer import style_transfer

send_image = None
history_mask = None
color_mask = None

detect_image = None
detect_image_ori = None
detect_label = None
index = None

x_pos, y_pos = 0, 0


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

    # 存储各个类别遮罩的颜色，初始化初始遮罩为可能的背景
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


def is_point_in_rectangle(point, rectangle):
    (x, y) = point
    # 获取矩形的对角点
    (x1, y1), (x2, y2) = rectangle
    # 确定最小和最大坐标
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    # 判断点是否在矩形内
    return x_min <= x <= x_max and y_min <= y <= y_max


# 找到鼠标区域所选择的矩形并画出来
def find_rectangle_for_point(image, evt: gr.SelectData):
    global detect_image, detect_label, index
    x, y = int(evt.index[0]), int(evt.index[1])
    point = (x, y)
    is_find = 0
    for i, rectangle in enumerate(detect_label):
        if is_point_in_rectangle(point, rectangle):
            is_find = 1
            index = i
            break

    if is_find != 0:
        x1 = int(detect_label[index][0][0])
        y1 = int(detect_label[index][0][1])
        x2 = int(detect_label[index][1][0])
        y2 = int(detect_label[index][1][1])
        w = x2 - x1
        h = y2 - y1
        rectangle_select = cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 1)
        return rectangle_select


# 根据输入的Json文件自动创建多个文本框以供微调
def create_textbox_slider(image):
    global detect_label
    x = image.shape[1]
    y = image.shape[0]
    num_boxes = len(detect_label)
    textbox_slider = []
    for i in range(num_boxes):
        with gr.Row() as row:
            # gr.Markdown(f"<center> 在这里修改第{i + 1}处的坐标:</center>")
            gr.Textbox(scale=1, label=f"{i + 1}处的坐标")
            gr.Slider(start=0, stop=x, step=0.01, value=(detect_label[i][1][1] + detect_label[i][2][1]) / 2)
            gr.Slider(start=0, stop=y, step=0.01, value=(detect_label[i][1][2] + detect_label[i][2][2]) / 2)
        textbox_slider.append(row)
    return textbox_slider


def create_ui():
    with (gr.Blocks() as ui):
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

                    # 图像叠加底图
                    background_image = gr.Image(tool="editor", visible=False, interactive=True, scale=1)

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

                        with gr.Tab("滤镜") as filters:
                            with gr.Row():
                                radio = gr.Radio(["原图", "锐利", "流年", "HDR", "反色", "美食", "冷艳", "单色"],
                                                 label="滤镜选择", info="请选择你感兴趣的滤镜")
                                # threshold = gr.Slider(minimum=0, maximum=100, step=0.01, value=10, visible=False)

                        with gr.Tab("马赛克") as mosaic:
                            with gr.Column():
                                weight = gr.Slider(minimum=0, maximum=30, step=1, value=10)
                                apply_button = gr.Button(value="应用效果")

                        with gr.Tab("图像叠加") as overlay_but:
                            with gr.Column():
                                overlay = gr.Image(label="Overlay", tool="color-sketch", visible=True)
                                x_pos1 = gr.Slider(label="x坐标", minimum=0, maximum=512, step=1, value=0,
                                                   interactive=True)
                                y_pos1 = gr.Slider(label="y坐标", minimum=0, maximum=512, step=1, value=0,
                                                   interactive=True)
                                scale = gr.Slider(label="缩放倍数", minimum=0.2, maximum=5, value=1)
                                rotating_angle = gr.Slider(label="旋转角度", minimum=-180, maximum=180, step=0.1,
                                                           value=0,
                                                           interactive=True)
                                opacity = gr.Slider(label="透明度", minimum=0, maximum=255, step=0.1, value=255)
                                refresh = gr.Button(value="刷新")

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
                def update_edit_interface(img):
                    return img, gr.update(visible=True), gr.update(visible=False), gr.update(
                        visible=False), gr.update(
                        visible=False)

                # 按下单色按钮，更新交互界面
                def update_monochrome_interface(filter_type, img):
                    if filter_type == '单色':
                        return img, gr.update(visible=False), gr.update(visible=True), gr.update(
                            visible=False)  # 原图框，单色图框，阈值滑动条框
                    else:
                        return img, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

                # 按下马赛克按钮，更新界面

                def update_mosaic_interface(img):
                    return img, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

                # 更新界面
                def update_filters(img):
                    return img, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

                def overlay_update(img):
                    return img, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

                # 监听参数调整
                setup_listeners(ori_image, brightness, Contrast, Saturation, Sharpness, Temperature, output_edit_image)

                # 界面更新
                edit.select(fn=update_edit_interface, inputs=[ori_image],
                            outputs=[ori_image, ori_image, get_color_image, mosaic_image, background_image])
                radio.change(fn=update_monochrome_interface, inputs=[radio, ori_image],
                             outputs=[get_color_image, ori_image, get_color_image, mosaic_image])
                filters.select(fn=update_filters, inputs=[ori_image],
                               outputs=[ori_image, mosaic_image, background_image])
                mosaic.select(fn=update_mosaic_interface, inputs=[ori_image],
                              outputs=[mosaic_image, ori_image, mosaic_image, background_image])
                overlay_but.select(fn=overlay_update, inputs=[ori_image],
                                   outputs=[background_image, background_image, ori_image, mosaic_image])

                # 应用单色滤镜
                get_color_image.select(fn=apply_monochrome_filters, inputs=[get_color_image], outputs=output_edit_image)
                # 应用滤镜
                radio.change(fn=filter_process, inputs=[ori_image, radio], outputs=output_edit_image)

                apply_button.click(fn=apply_mosaic, inputs=[mosaic_image, weight], outputs=output_edit_image)

                def get_rotated_dimensions(width, height, angle, scale):
                    """根据旋转角度计算图像旋转后的尺寸"""
                    # radians = math.radians(angle)
                    # 计算旋转前的对角线长度
                    diagonal = math.sqrt(width ** 2 + height ** 2)
                    # 计算旋转后的尺寸（对角线长度乘以缩放因子）
                    new_width = new_height = diagonal * scale
                    # 返回计算结果
                    return int(new_width), int(new_height)

                # 图像旋转叠加
                # def overlay_images(ori_img, overlay_image, scale, rotating_angle, opacity, evt: gr.SelectData):
                def overlay_images(ori_img, overlay_image, x, y, scale, rotating_angle, opacity):
                    # global x_pos, y_pos
                    # x, y = int(evt.index[0]), int(evt.index[1])
                    #
                    # if evt.index is not None:  # 添加这个检查
                    #     x, y = int(evt.index[0]), int(evt.index[1])
                    #     if x_pos != x or y_pos != y:
                    #         x_pos = x
                    #         y_pos = y
                    # else:
                    #     # 如果 evt.index 是 None，使用之前的 x_pos 和 y_pos
                    #     x = x_pos
                    #     y = y_pos

                    width, height = ori_img.shape[:2]
                    width1, height1 = overlay_image.shape[:2]
                    rotated_matrix = cv2.getRotationMatrix2D((width1 / 2, height1 / 2), rotating_angle, scale)
                    new_width, new_height = get_rotated_dimensions(width1, height1, rotating_angle, scale)
                    rotated_image = cv2.warpAffine(overlay_image, rotated_matrix, (new_width, new_height))

                    # 超出部分计算
                    if x + new_width > width:
                        fw = ori_img.shape[1]
                    else:
                        fw = x + new_width
                    if y + new_height > height:
                        fh = ori_img.shape[0]
                    else:
                        fh = y + new_height

                    rotated_image = rotated_image[:fh - y, :fw - x]

                    # 创建结果图像的副本
                    result_image = ori_img.copy()
                    # 确定叠加区域
                    overlay_area = result_image[y:fh, x:fw]

                    # 分离前景图像的RGB和alpha
                    if overlay_image.shape[2] == 4:
                        alpha_channel = overlay_area[:, :, 3]
                        rgb_channel = overlay_area[:, :, :3]
                    else:
                        alpha_channel = np.ones((fh - y, fw - x), dtype=np.uint8) * 255
                        rgb_channel = rotated_image[:, :, :3]

                    black_mask = np.all(rgb_channel == [0, 0, 0], axis=-1)
                    alpha_channel[black_mask == True] = 0
                    alpha_channel[~black_mask == True] = 255 - (255 - opacity)
                    # 创建掩码，使用alpha通道确定透明部分
                    mask = alpha_channel / 255.0

                    for c in range(0, 3):
                        overlay_area[:, :, c] = (1.0 - mask) * overlay_area[:, :, c] + mask * rgb_channel[:, :, c]

                    result_image[y:fh, x:fw, :] = overlay_area
                    return result_image

                # background_image.select(fn=overlay_images,
                #                         inputs=[background_image, overlay, scale, rotating_angle, opacity],
                #                         outputs=output_edit_image)

                def update_xy_slider(image):
                    x_length, y_length = image.shape[1], image.shape[0]
                    return gr.update(maximum=x_length), gr.update(maximum=y_length)

                refresh.click(fn=update_xy_slider, inputs=[background_image], outputs=[x_pos1, y_pos1])

                x_pos1.change(fn=overlay_images,
                              inputs=[background_image, overlay, x_pos1, y_pos1, scale, rotating_angle, opacity],
                              outputs=output_edit_image)
                y_pos1.change(fn=overlay_images,
                              inputs=[background_image, overlay, x_pos1, y_pos1, scale, rotating_angle, opacity],
                              outputs=output_edit_image)
                scale.change(fn=overlay_images,
                             inputs=[background_image, overlay, x_pos1, y_pos1, scale, rotating_angle, opacity],
                             outputs=output_edit_image)
                rotating_angle.change(fn=overlay_images,
                                      inputs=[background_image, overlay, x_pos1, y_pos1, scale, rotating_angle,
                                              opacity],
                                      outputs=output_edit_image)
                opacity.change(fn=overlay_images,
                               inputs=[background_image, overlay, x_pos1, y_pos1, scale, rotating_angle, opacity],
                               outputs=output_edit_image)

        with gr.Tab("美颜"):
            with gr.Row():
                with gr.Column():
                    face_image = gr.Image()
                    with gr.Tab("参数调节"):
                        big_eyes = gr.Slider(label="大眼", minimum=0, maximum=50, step=0.1, value=20,
                                             interactive=True)
                        whitening = gr.Slider(label="美白", minimum=0, maximum=100, step=0.1, value=30,
                                              interactive=True)
                        smooth = gr.Slider(label="磨皮", minimum=0, maximum=1, step=0.01, value=0.3,
                                           interactive=True)
                        thin_face = gr.Slider(label="瘦脸", minimum=0.8, maximum=1.2, step=0.01, value=1,
                                              interactive=True)

                with gr.Column(scale=1):
                    beauty_image = gr.Image()

        big_eyes.change(fn=beauty_image_processing, inputs=[face_image, big_eyes, whitening, smooth, thin_face],
                        outputs=[beauty_image])
        whitening.change(fn=beauty_image_processing,
                         inputs=[face_image, big_eyes, whitening, smooth, thin_face],
                         outputs=[beauty_image])
        smooth.change(fn=beauty_image_processing,
                      inputs=[face_image, big_eyes, whitening, smooth, thin_face],
                      outputs=[beauty_image])
        thin_face.change(fn=beauty_image_processing,
                         inputs=[face_image, big_eyes, whitening, smooth, thin_face],
                         outputs=[beauty_image])

        #抠图设置
        with gr.Tab("抠图"):
            with gr.Row():
                with gr.Column():
                    #
                    # 点击交互式抠图的按钮后，隐藏img_cutout，显示img_cutout_inter，可进行交互抠图。提醒：笔刷颜色需要是黑色/白色
                    ori_cutout_img = gr.Image(label="Output image", interactive=True, visible=True)
                    img_cutout_interactive = gr.Image(label="Background", source="upload", tool="sketch", type="pil",
                                                      brush_color='#42b983',
                                                      mask_opacity=0.5, brush_radius=100, visible=False,
                                                      interactive=True, scale=1)

                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                interactive_button = gr.Button(value="交互式抠图", visible=True)
                                auto_cutout = gr.Button(value="自动抠图")
                                # photo_make = gr.Button(value="证件照制作", visible=True)
                            exit_button = gr.Button(value="退出交互抠图模式", visible=False)
                        with gr.Column():
                            # 前景: (0, 255, 0)，背景:(255,0,0)，可能前景:(0, 255, 255)，可能背景:(255, 0, 255)
                            select_mask_mode = gr.Radio(['前景', '背景', '可能前景', '可能背景'], visible=False,
                                                        label="请选择mask属性", interactive=True)
                            with gr.Row():
                                ensure_mask = gr.Button(value="确定遮罩选择", visible=False, interactive=True)
                                ensure_button = gr.Button(value="确定", visible=False, interactive=True)

                with gr.Column(scale=1):
                    cutout_color = gr.Image(label="Output image", interactive=False)
                    cutout_result = gr.Image(label="Output image", interactive=False, visible=False)
                    with gr.Row():
                        clear_mask_but = gr.Button(value="清除遮罩", visible=False, interactive=True)
                        gr.Button(value="Save")

                # 图片,交互式抠图显示框,原抠图显示框 退出按钮,select_mask_mode,ensure_button,ensure_mask,自动抠图，证件照制作, cutout_result,clear_mask_but
                def update_cutout_interface(img):
                    return img, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(
                        visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
                        visible=False), gr.update(visible=True), gr.update(visible=True)

                def recover_cutout_interface():
                    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(
                        visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
                        visible=True), gr.update(visible=False), gr.update(visible=False)

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

                interactive_button.click(fn=update_cutout_interface, inputs=[ori_cutout_img],
                                         outputs=[img_cutout_interactive, ori_cutout_img, img_cutout_interactive,
                                                  exit_button, select_mask_mode,
                                                  ensure_button, ensure_mask,
                                                  auto_cutout, cutout_result, clear_mask_but])
                exit_button.click(fn=recover_cutout_interface,
                                  outputs=[ori_cutout_img, img_cutout_interactive, exit_button, select_mask_mode,
                                           ensure_button, ensure_mask,
                                           auto_cutout, cutout_result, clear_mask_but])
                select_mask_mode.change(fn=update_mask_mode, inputs=[select_mask_mode], outputs=img_cutout_interactive)

                ensure_mask.click(fn=get_mask, inputs=[img_cutout_interactive, select_mask_mode],
                                  outputs=[cutout_color])
                clear_mask_but.click(fn=clear_mask, outputs=[cutout_color])
                global history_mask, color_mask
                ensure_button.click(fn=get_mask_image, inputs=[img_cutout_interactive],
                                    outputs=[cutout_result])

        with gr.Tab("人物识别"):
            with gr.Row():
                with gr.Column():
                    det_image = gr.Image(label="输入待识别的图片", interactive=True)
                    # detect_threshold = gr.Slider(label="检测阈值", minimum=0, maximum=1, value=0.5, step=0.01)
                    detect_but = gr.Button(value="检测", interactive=True)
                    with gr.Accordion("坐标微调", open=False) as accordion:
                        select_rect_img = gr.Image(interactive=False)
                        slider1 = gr.Slider(label="左上角x值", minimum=0, maximum=512, step=1, value=128,
                                            interactive=True)
                        slider2 = gr.Slider(label="左上角y值", minimum=0, maximum=512, step=1, value=128,
                                            interactive=True)
                        slider3 = gr.Slider(label="右下角x值", minimum=0, maximum=512, step=1, value=384,
                                            interactive=True)
                        slider4 = gr.Slider(label="右下角y值", minimum=0, maximum=512, step=1, value=384,
                                            interactive=True)
                        sliders = [slider1, slider2, slider3, slider4]
                        save_but = gr.Button(value="保存改动")

                with gr.Column(scale=1):
                    global detect_label
                    detect_result_image = gr.Image(label="识别结果", interactive=True, tool='editor', scale=3)
                    detect_label = gr.Textbox()

                def detect_result(image):
                    global detect_image, detect_label, detect_image_ori
                    detect_image_ori = image
                    detect_label = get_prediction(image)
                    image = draw_process(image, detect_label)
                    detect_image = image
                    return image, detect_label

                def update_sliders(image, evt: gr.SelectData):
                    global detect_label, index
                    rect_img = find_rectangle_for_point(image, evt)

                    x1 = detect_label[index][0][0]
                    y1 = detect_label[index][0][1]
                    x2 = detect_label[index][1][0]
                    y2 = detect_label[index][1][1]

                    # 确保所有值都是整数
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # 扩展像素
                    width = image.shape[1]
                    height = image.shape[0]
                    x1_expanded = max(0, x1 - 30)
                    y1_expanded = max(0, y1 - 30)
                    x2_expanded = min(width, x2 + 30)
                    y2_expanded = min(height, y2 + 30)

                    cropped_image = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

                    # 选出矩形的图像， 更新左上角x,y值，右下角x,y值
                    return cropped_image, gr.Slider.update(minimum=x1 - 30, maximum=x1 + 30,
                                                           value=x1), gr.Slider.update(minimum=y1 - 30, maximum=y1 + 30,
                                                                                       value=y1), gr.Slider.update(
                        minimum=x2 - 30, maximum=x2 + 30, value=x2), gr.Slider.update(minimum=y2 - 30, maximum=y2 + 30,
                                                                                      value=y2)

                def update_rector(x1, y1, x2, y2, image):
                    width = image.shape[1]
                    height = image.shape[0]
                    x1_expanded = max(0, x1 - 30)
                    y1_expanded = max(0, y1 - 30)
                    x2_expanded = min(width, x2 + 30)
                    y2_expanded = min(height, y2 + 30)
                    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cropped_image = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
                    return cropped_image

                def save_update(x1, y1, x2, y2):
                    global detect_label, index, detect_image_ori
                    detect_label[index][0][0] = x1
                    detect_label[index][0][1] = y1
                    detect_label[index][1][0] = x2
                    detect_label[index][1][1] = y2
                    image = draw_process(detect_image_ori, detect_label)
                    return image, detect_label

                detect_but.click(fn=detect_result, inputs=[det_image], outputs=[detect_result_image, detect_label])
                detect_result_image.select(fn=update_sliders, inputs=[det_image],
                                           outputs=[select_rect_img, slider1, slider2, slider3, slider4])

                for slider in sliders:
                    slider.change(fn=update_rector, inputs=[slider1, slider2, slider3, slider4, det_image],
                                  outputs=[select_rect_img])
                save_but.click(fn=save_update, inputs=[slider1, slider2, slider3, slider4],
                               outputs=[detect_result_image, detect_label])

        with gr.Tab("发现"):
            with gr.Tab("风格迁移"):
                with gr.Row():
                    with gr.Column():
                        ori_image = gr.Image()
                        choice = ["candy", "composition_vii", "feathers", "la_muse", "mosaic", "starry_night",
                                  "the_wave", "udnie"]
                        style_select = gr.Dropdown(choices=choice, label="style")

                    with gr.Column(scale=1):
                        output_image = gr.Image()

                style_select.change(fn=style_transfer, inputs=[ori_image, style_select], outputs=[output_image])

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

            with gr.Tab("图像修复"):
                with gr.Column(scale=3):
                    input_image = gr.Image(label="Background", source="upload", tool="sketch", type="pil",
                                           brush_color='#000000',
                                           mask_opacity=0.5, brush_radius=100, interactive=True, height=512)
                    sure_but = gr.Button(value="确定")
                with gr.Column(scale=1):
                    output_image = gr.Image()
                sure_but.click(fn=Remove_watermark, inputs=[input_image], outputs=[output_image])

    return ui
