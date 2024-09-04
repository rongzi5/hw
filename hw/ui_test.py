import gradio as gr

with gr.Blocks() as demo:
    base = gr.Image(label="Background", source="upload", tool="sketch", type="pil", height=512, brush_color='#42b983', mask_opacity=0.5, brush_radius=100)
    text = gr.Image(label="Image")
    button = gr.Button()


    def f(img):

        mask = img["mask"]
        return mask


    button.click(fn=f,inputs=base, outputs=[text])

demo.launch()

base = gr.Image(label="Background", source="upload", tool="color-sketch", type="pil", height=512, brush_color='#42b983',
                mask_opacity=0.5, brush_radius=100)