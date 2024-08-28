import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from torchvision import transforms

carrier_image = Image.open('carrier.jpg')
secret_image = Image.open('secret1.jpg')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
carrier_tensor = transform(carrier_image)
secret_tensor = transform(secret_image)
H = carrier_tensor.shape[1]
W = carrier_tensor.shape[2]
new_tensor = torch.zeros(3 * H * W)

secret_flat = secret_tensor.view(-1)
carrier_flat = carrier_tensor.view(-1)

for i in range(secret_flat.numel()):
    secret_bit = secret_flat[i] % 2
    new_tensor[i] = (carrier_flat[i] // 2) + secret_bit

new_tensor = new_tensor.view(3, H, W)

encrypted_flat = new_tensor.view(-1)
new_secret_tensor = torch.zeros(3 * H * W)

for i in range(H * W * 3):
    new_secret_tensor[i] = encrypted_flat[i] % 2
new_secret_tensor = new_secret_tensor.view(3, H, W)

tensor_list = [carrier_tensor, secret_tensor, new_tensor, new_secret_tensor]
tensor_list=[s.float() for s in tensor_list]
title = ['carrier images', 'secret images', 'Encrypt images', 'Restored pictures']

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i, (tensor, ax) in enumerate(zip(tensor_list, axs.ravel())):
    ax.imshow(tensor.permute(1, 2, 0).numpy())
    ax.axis('off')
    ax.set_title(title[i])
plt.tight_layout()
plt.show()
