import torch, torchvision
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from ANN import ANN

def train_per_epoch():
    current_loss = 0.0
    last_loss = 0.0
    for i, data in enumerate(train_loader):
        imgs, labels = data  # 获取一批训练数据; imgs(64, 1, 28, 28)
        # 1.更新参数的梯度清零
        optimizer.zero_grad()
        # 2.输入X->正向传播输出y^ -> 预测值
        predicts = model(imgs.view(-1, 28*28))
        # 3.计算预测值和真值label的差别 -> J
        loss = loss_fn(predicts, labels)
        # 4.反向传播 -> 计算梯度向量
        loss.backward()
        # 5.更新参数 -> w, b
        optimizer.step()
        current_loss += loss.item()  # 累加损失
        # 每100批计算平均损失, 输出打印......
        if i % 100 == 99:
            last_loss = current_loss / 100
            print('   batch {} loss: {}'.format(i+1, last_loss))
            current_loss = 0.0

    return last_loss

def valuate():
    corrects = 0
    for data, label in test_loader:
        with torch.no_grad():
            logits = model(data.view(-1, 28*28))
            pred = logits.argmax(dim=1)
        corrects += torch.eq(pred, label).sum().float().item()
    return corrects / len(test_dataset)



# 下载/加载内置数据集FashionMNIST; map-style数据集
train_dataset = FashionMNIST(root='./data',
                            train=True,
                            download=True,
                            transform=transforms.ToTensor())
test_dataset = FashionMNIST(root='./data',
                            train=False,
                            download=True,
                            transform=transforms.ToTensor())

# 打印数据集的特征
# print(type(train_dataset), len(train_dataset), len(test_dataset))

# 创建训练和测试数据的加载器对象DataLoader
batch_size = 64  # 每次加载多少个数据到内存中, out of memory内存溢出, 建议减少改值32/16/8
train_loader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True)
test_loader = DataLoader(test_dataset,
                        batch_size=batch_size,
                        shuffle=True)


model = ANN()

# 损失函数: 衡量预测值和真值/标签的差别(距离differences) -> 用于多分类问题
loss_fn = nn.CrossEntropyLoss()
epochs = 20

# 优化器算法/寻找相对极小值
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(epochs):
    print(f'EPOCH --> {epoch + 1} ....')
    # 将最后一批数据的平均损失打印
    last_avg_loss = train_per_epoch()
    print('last average loss is ---> ', last_avg_loss)
    acc = valuate()
    print(f'The accuracy is ----> {acc}')



