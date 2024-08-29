from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

#提取隐写函数图像
def to_stego_image(carrier_image, secret_image):
    # 读取图像
    carrier_image = carrier_image.convert('RGB')
    secret_image = secret_image.convert('RGB')

    # 转换图像为Tensor，并保持0-255范围
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
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


def extract_secret_image(stego_image):
    # 转换图像为Tensor，并保持0-255范围
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为[0, 1]范围
        transforms.Lambda(lambda x: (x * 255).byte())  # 还原到[0, 255]范围并转换为整数
    ])
    stego_tensor = transform(stego_image)

    # 获取图像的高度和宽度
    H, W = stego_tensor.shape[1], stego_tensor.shape[2]

    # 将Tensor展平为一维
    stego_flat = stego_tensor.view(-1).int()  # 转换为整数

    # 提取秘密图像的最低三位，并移位回原位置
    extracted_secret_flat = (stego_flat & 0x07) << 5  # 提取最低三位并移位回原位置

    # 重新调整Tensor形状并转换为byte
    extracted_secret_tensor = extracted_secret_flat.view(3, H, W).byte()

    # 将Tensor转换回PIL图像
    to_pil = transforms.ToPILImage()
    secret_image = to_pil(extracted_secret_tensor)

    return secret_image
