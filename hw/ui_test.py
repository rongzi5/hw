import gradio as gr

with gr.Blocks() as demo:
    base = gr.Image(label="Background", source="upload", tool="sketch", type="pil", height=512, brush_color='#FFFFFF', mask_opacity=0.5, brush_radius=100)
    image = gr.Image(label="Image")

    @gr.render(input=base, output=image)
    def f(img):
        # 获取 img 字典中的 mask 图像并转换为灰度图像
        mask = img["mask"]
        if mask is not None:
            return mask.convert("L")  # 转换为灰度图像
        else:
            return None  # 如果没有遮罩，返回 None

demo.launch()
