import torch
from torch import nn
from torch.nn import functional as F


class Lenet5(nn.Module):
	"""
		for cifar10
	"""

	def __init__(self):
		super(Lenet5, self).__init__()

		self.conv_unit = nn.Sequential(
			nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
			nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
			nn.Flatten()
		)

		self.fc_unit = nn.Sequential(
			nn.Linear(16 * 5 * 5, 120),
			nn.ReLU(),
			nn.Linear(120, 84),
			nn.ReLU(),
			nn.Linear(84, 10)
		)

		self.criteon = nn.CrossEntropyLoss()

	def forward(self, x):
		x = self.conv_unit(x)
		logits = self.fc_unit(x)

		return logits

def main():

	net = Lenet5()
	tmp = torch.randn(2,3,32,32)
	out = net(tmp)
	print(out.shape)


if __name__ == '__main__':
	main()