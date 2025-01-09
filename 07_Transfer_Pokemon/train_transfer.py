import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from pokemon import Pokemon
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

batches = 64
lr = 1e-3
epochs = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)

train_db = Pokemon('data/pokemon', 224, mode='train')
val_db = Pokemon('data/pokemon', 224, mode='val')
test_db = Pokemon('data/pokemon', 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batches, shuffle=True, num_workers=8)
val_loader = DataLoader(val_db, batch_size=batches, num_workers=4)
test_loader = DataLoader(test_db, batch_size=batches, num_workers=4)

def evaluate(model, loader):
	corrects = 0
	total = len(loader.dataset)

	for x, y in loader:
		x, y = x.to(device), y.to(device)
		with torch.no_grad():
			logi = model(x)
			pred = logi.argmax(dim=1)
		corrects += torch.eq(pred, y).sum().float().item()

	return corrects / total


def main():

	trained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
	model = nn.Sequential(*list(trained_model.children())[:-1],
	                      nn.Flatten(),
	                      nn.Linear(512, 5)
	                      ).to(device)

	optimizer = optim.Adam(model.parameters(), lr=lr)
	criteon = nn.CrossEntropyLoss().to(device)
	best_acc = 0
	best_epoch = 0
	
	for epoch in range(epochs):

		print(f"Epoch {epoch+1} start Training...")
		for step, (x, y) in enumerate(tqdm(train_loader)):
			x, y = x.to(device), y.to(device)
			logits = model(x)
			loss = criteon(logits, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
	

		if epoch % 2 == 0:
			val_acc = evaluate(model, val_loader)
			print("epoch:", epoch+1, "val acc:", val_acc)
			if val_acc > best_acc:
				best_epoch = epoch
				best_acc = val_acc

				torch.save(model.state_dict(), '07_Transfer_Pokemon/log_model/best.mdl')

	print("best acc:", best_acc, "best epoch:", best_epoch)

	model.load_state_dict(torch.load("07_Transfer_Pokemon/log_model/best.mdl"))
	print("loaded from ckpt!")

	test_acc = evaluate(model, test_loader)
	print("test acc:", test_acc)


if __name__ == '__main__':
	main()