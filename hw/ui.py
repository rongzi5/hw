import gradio as gr
from ui.ui import create_ui

if __name__ == '__main__':
    gr_ui = create_ui()
    gr_ui.launch()
