import gradio as gr


def create_html(image_path, text):
    # 使用Base64编码图像，以便在HTML中直接使用
    import base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    image_src = f"data:image/png;base64,{encoded_string}"

    # 返回包含图像和可拖动文本的HTML内容
    html_content = f"""
    <html>
    <head>
        <style>
            #canvas {{ cursor: move; border: 1px solid black; }}
            body {{ display: flex; justify-content: center; align-items: center; height: 100vh; }}
        </style>
    </head>
    <body>
    <canvas id="canvas"></canvas>
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const image = new Image();
        image.src = "{image_src}";
        image.onload = () => {{
            canvas.width = image.width;
            canvas.height = image.height;
            ctx.drawImage(image, 0, 0);
            ctx.fillText("{text}", 100, 100); // 初始位置
        }};

        let drag = false;
        let offsetX = 100;
        let offsetY = 100;

        canvas.addEventListener('mousedown', function(e) {{
            drag = true;
        }});

        canvas.addEventListener('mousemove', function(e) {{
            if (drag) {{
                offsetX = e.offsetX;
                offsetY = e.offsetY;
                redraw();
            }}
        }});

        canvas.addEventListener('mouseup', function() {{
            drag = false;
        }});

        function redraw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(image, 0, 0);
            ctx.fillText("{text}", offsetX, offsetY);
        }}
    </script>
    </body>
    </html>
    """
    return html_content


iface = gr.Interface(
    fn=create_html,
    inputs=[gr.inputs.Image(type="filepath"), gr.inputs.Textbox(default="Drag me!")],
    outputs="html"
)

iface.launch()
