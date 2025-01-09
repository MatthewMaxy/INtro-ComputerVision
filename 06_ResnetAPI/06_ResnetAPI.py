import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# 加载预训练的ResNet-18模型
model = models.resnet18(pretrained=True)
model.eval()

# 图片预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# url1:金鱼 ； url2: 蛇

# url = 'https://imagepphcloud.thepaper.cn/pph/image/262/982/946.jpg'  # 替换为实际图片URL
url = 'https://s.yimg.com/ny/api/res/1.2/IRdN7bOyTLZqwzrgYU4i3g--/YXBwaWQ9aGlnaGxhbmRlcjt3PTgwMDtoPTUzNg--/https://media.zenfs.com/zh-Hant-TW/homerun/mirrormedia.mg/c31d95a92a41702755539d0a051a27b0'  # 替换为实际图片URL
response = requests.get(url)
img = Image.open(BytesIO(response.content))

img_tensor = transform(img).unsqueeze(0)  # 添加批次维度

# 进行预测
with torch.no_grad():
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)

print(predicted.item())