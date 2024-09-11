from .dayan import update_image_with_slider_bigeyes
from .meibai import process_image_with_slider_white
from .mopi import process_image_with_slider_smooth
from .shoulian import face_thin_auto
from .mopi2 import process_image_with_fixed_sigma_s


def beauty_image_processing(image, bigeyes_value, white_value, smooth_value, thin_value):
    # 大眼处理
    image = update_image_with_slider_bigeyes(image, bigeyes_value)

    # 美白处理
    image = process_image_with_slider_white(image, white_value)

    # 磨皮处理
    image = process_image_with_fixed_sigma_s(image, smooth_value)

    # 瘦脸处理
    image = face_thin_auto(image, thin_value)

    return image
