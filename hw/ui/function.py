from PIL import Image
import cv2
import numpy as np
from torchvision import transforms
from .utils import img_resize
import gradio as gr


#隐写函数图像
def to_stego_image(carrier_image, secret_image):
    flag, carrier_image, secret_image = img_resize(carrier_image, secret_image)
    # 读取图像
    # 如果输入是NumPy数组，先转换为PIL图像
    if flag:
        gr.Info("图片尺寸不同，将会影响最终效果")
    if isinstance(carrier_image, np.ndarray):
        carrier_image = Image.fromarray(carrier_image)
    if isinstance(secret_image, np.ndarray):
        secret_image = Image.fromarray(secret_image)
    # 转换图像为Tensor，并保持0-255范围
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为[0, 1]范围
        transforms.Lambda(lambda x: (x * 255).byte())  # 还原到[0, 255]范围并转换为整数
    ])
    carrier_tensor = transform(carrier_image)
    secret_tensor = transform(secret_image)

    # 获取图像的高度和宽度
    H = carrier_tensor.shape[1]
    W = carrier_tensor.shape[2]

    # 将Tensor展平为一维
    carrier_flat = carrier_tensor.view(-1).int()  # 转换为整数
    secret_flat = secret_tensor.view(-1).int()

    # 提取秘密图像的最高三位
    secret_high_bits = (secret_flat >> 5) & 0x07  # 最高三位，掩码0x07为三位

    # 将载体图像的最低三位清零
    carrier_low_bits_cleared = carrier_flat & 0xF8  # 掩码0xF8为保留高五位，清除最低三位

    # 将秘密图像的最高三位嵌入载体图像的最低三位
    new_tensor_flat = carrier_low_bits_cleared | secret_high_bits

    # 将一维数据恢复为图像的原始形状
    new_tensor = new_tensor_flat.view(3, H, W).byte()  # 恢复为3通道的图像
    # 将Tensor转换回PIL图像
    to_pil = transforms.ToPILImage()
    stego_image = to_pil(new_tensor)

    return stego_image


#提取隐写图像
def extract_secret_image(stego_image):
    # 转换图像为Tensor，并保持0-255范围
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为[0, 1]范围
        transforms.Lambda(lambda x: (x * 255).byte())  # 还原到[0, 255]范围并转换为整数
    ])
    stego_tensor = transform(stego_image)

    # 获取图像的高度和宽度
    H, W = stego_tensor.shape[1], stego_tensor.shape[2]
    stego_flat = stego_tensor.view(-1).int()  # 转换为整数

    # 提取秘密图像的最低三位，并移位回原位置
    extracted_secret_flat = (stego_flat & 0x07) << 5  # 提取最低三位并移位回原位置
    extracted_secret_tensor = extracted_secret_flat.view(3, H, W).byte()

    # 将Tensor转换回PIL图像
    to_pil = transforms.ToPILImage()
    secret_image = to_pil(extracted_secret_tensor)

    # 提取载体图像的高五位
    carrier_flat = (stego_flat & 0xF8)  # 保留高五位

    # 重新调整Tensor形状并转换为byte
    carrier_tensor = carrier_flat.view(3, H, W).byte()
    carrier_image = to_pil(carrier_tensor)

    return carrier_image, secret_image


# 编辑图像参数(对比度/亮度/饱和度等)
# 亮度和对比度
def edit_bright_contrast(image, bright, contrast):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=bright)


# 色温
def edit_temperature(img, temperature):
    # 确保图像是uint8类型
    if img.dtype != np.uint8:
        img = np.uint8(img)

    R, G, B = cv2.split(img)
    R = np.clip(R + temperature, 0, 255).astype(np.uint8)
    G = np.clip(G + temperature, 0, 255).astype(np.uint8)
    B = np.clip(B - temperature, 0, 255).astype(np.uint8)
    return cv2.merge((R, G, B))


# 饱和度
def Saturation(rgb_img, saturation):
    img_t = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_t)
    s = np.clip(s * saturation, 0, 255).astype(np.uint8)  # 确保数据类型为 uint8
    img_t = cv2.merge((h, s, v))  # 确保合并时数据类型一致
    img = cv2.cvtColor(img_t, cv2.COLOR_HSV2BGR)
    return img


# 锐化
def laplacian_sharpening(image, sharpness, kernel_size=3, scale=0.03, delta=0):
    # Convert the image to grayscale
    image = np.uint8(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=kernel_size, scale=scale, delta=delta)

    # Convert Laplacian image to 8-bit
    laplacian_8u = cv2.convertScaleAbs(laplacian)

    # Convert the 8-bit Laplacian image to BGR
    laplacian_8u_bgr = cv2.cvtColor(laplacian_8u, cv2.COLOR_GRAY2BGR)

    # Add the Laplacian image to the original image
    sharpened_image = cv2.addWeighted(image, 1, laplacian_8u_bgr, sharpness, 0)

    return sharpened_image


# 编辑图像
def edit_img(img, brightness=0.0, contrast=1.0, saturation=1.0, sharpness=0.0, temperature=0.0):
    # 亮度和对比度调整
    img = edit_bright_contrast(img, brightness, contrast)
    # 饱和度调整
    img = Saturation(img, saturation)
    # 锐度调整
    img = laplacian_sharpening(img, sharpness)
    # 色温调整
    img = edit_temperature(img, temperature)

    return img


# 图像滤镜处理
def filter_process(image, index):
    # 锐利效果
    def sharpen(img):
        kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        img_sharpen = cv2.filter2D(img, -1, kernel)
        return img_sharpen

    # 棕褐色滤镜
    def sepia(img):
        # 将图像转换为浮点数，以避免数据丢失
        img_sepia = np.array(img, dtype=np.float32)
        # 进行矩阵变换，应用Sepia滤镜
        sepia_matrix = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        img_sepia = cv2.transform(img_sepia, sepia_matrix)
        # 将超出255的值剪裁到255，确保所有值都在0-255之间
        img_sepia = np.clip(img_sepia, 0, 255)
        # 将图像转换为uint8类型
        img_sepia = np.array(img_sepia, dtype=np.uint8)
        return img_sepia

    # HDR effect
    def HDR(img):
        hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
        return hdr

    # 反转滤镜
    def invert(img):
        inv = cv2.bitwise_not(img)
        return inv

    # 美食滤镜
    def delicious_food(img):
        img1 = cv2.convertScaleAbs(img, alpha=1, beta=-10)
        img2 = Saturation(img1, saturation=1.3)
        img3 = laplacian_sharpening(img2, sharpness=1.5)
        return img3

    # 冷艳滤镜
    def cold_filter(img):
        pass

