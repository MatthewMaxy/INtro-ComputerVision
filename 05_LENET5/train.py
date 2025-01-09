import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from Lenet5 import Lenet5
from torch import nn, optim
from tqdm import tqdm

def main():
	
	cifar_train = DataLoader(
		datasets.CIFAR10('./data/cifar', True, download=True,
		                 transform=transforms.Compose([
			                 transforms.Resize((32, 32)),
			                 transforms.ToTensor()
		                 ])),
		batch_size=32, shuffle=True)

	cifar_test = DataLoader(
		datasets.CIFAR10('./data/cifar', False, download=True,
		                 transform=transforms.Compose([
			                 transforms.Resize((32, 32)),
			                 transforms.ToTensor()
		                 ])),
		batch_size=32, shuffle=True
	)

	# device = torch.device('cuda')
	# model = Lenet5().to(device)
	# criteon = nn.CrossEntropyLoss().to(device)

	model = Lenet5()
	criteon = nn.CrossEntropyLoss()
	
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	for epoch in range(20):

		model.train()
		for x, label in tqdm(cifar_train):
			# x, label = x.to(device), label.to(device)
			
			logits = model(x)
			loss = criteon(logits, label)

			# backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print(f"Epoch:{epoch}, loss is{loss.item()}")
        
		model.eval()
		# test
		with torch.no_grad():
			total_correct = 0
			total_num = 0
			for x, label in tqdm(cifar_test):
				
				# x, label = x.to(device), label.to(device)
				logits = model(x)
				pred = logits.argmax(dim=1)
				total_correct += torch.eq(pred, label).float().sum().item()
				total_num += x.size(0)

			acc = total_correct/total_num
			print(f"Epoch:{epoch}: The accuracy is {acc}")


if __name__ == '__main__':
	main()