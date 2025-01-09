import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from pokemon import Pokemon
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np
batches = 64
lr = 1e-3
epochs = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# 加载数据集
train_db = Pokemon('data/pokemon', 224, mode='train')
val_db = Pokemon('data/pokemon', 224, mode='val')
test_db = Pokemon('data/pokemon', 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batches, shuffle=True, num_workers=8)
val_loader = DataLoader(val_db, batch_size=batches, num_workers=4)
test_loader = DataLoader(test_db, batch_size=batches, num_workers=4)

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.config.classifier = 'mlp'
model.config.num_labels = 5
model.classifier = nn.Linear(768,5)
parameters = list(model.parameters())

for x in parameters[:-1]:
    x.requires_grad = False

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# print(model)
# 评估函数
def evaluate(model, loader):
    corrects = 0
    total = len(loader.dataset)

    for inputs, label in loader:
        
        inputs = processor(images=inputs, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to(device)
        label = label.to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            pred = logits.argmax(dim=1)
        corrects += torch.eq(pred, label).sum().float().item()

    return corrects / total

# 主训练函数
def main():
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss().to(device)

    best_acc = 0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        for step, (inputs, label) in enumerate(tqdm(train_loader)):
            inputs = processor(images=inputs, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].to(device)
            label = label.to(device)

            logits = model(**inputs).logits
            loss = criteon(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            model.eval()
            val_acc = evaluate(model, val_loader)
            print(f"epoch:{epoch}, loss: {loss}, val_acc:{val_acc}")
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_vit_model.mdl')

    print("best acc:", best_acc, "best epoch:", best_epoch)

    # 加载最优模型
    model.load_state_dict(torch.load("best_vit_model.mdl"))
    print("loaded from best checkpoint!")

    # 测试集评估（可以换成测试集）
    test_acc = evaluate(model, test_loader)
    print("test acc:", test_acc)

if __name__ == '__main__':
    main()
