import csv
import glob
import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Pokemon(Dataset):
	def __init__(self, root, resize, mode):
		super(Pokemon, self).__init__()

		self.root = root
		self.resize = resize

		self.name2label = {}
		for name in sorted(os.listdir(os.path.join(root))):
			if not os.path.isdir(os.path.join(root, name)):
				continue
			self.name2label[name] = len(self.name2label.keys())
		# print(self.name2label)
		self.images, self.labels = self.load_csv('images.csv')

		# train:60%     val:20%     test:20%
		if mode == 'train':
			self.images = self.images[:int(0.6*len(self.images))]
			self.labels = self.labels[:int(0.6*len(self.labels))]
		elif mode == 'val':
			self.images = self.images[int(0.6*len(self.images)):int(0.8*len(self.images))]
			self.labels = self.labels[int(0.6*len(self.labels)):int(0.8*len(self.labels))]
		else:
			self.images = self.images[int(0.8*len(self.images)):]
			self.labels = self.labels[int(0.8*len(self.labels)):]

	# image_path, label
	def load_csv(self, filename):
		if not os.path.exists(os.path.join(self.root, filename)):
			images = []
			for name in self.name2label.keys():
				images += glob.glob(os.path.join(self.root, name, '*.png'))
				images += glob.glob(os.path.join(self.root, name, '*.jpg'))
				images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

			# 1167, pokeman/bulbasaur/00000000.png, 0
			
			random.shuffle(images)
			with open(os.path.join(self.root, filename), mode='w', newline='') as f:
				writer = csv.writer(f)
				for img in images:
					name = img.split(os.sep)[-2]
					label = self.name2label[name]
					writer.writerow([img, label])
				print('writen into csv file: ', filename)

		# read from csv
		images, labels = [], []
		with open(os.path.join(self.root, filename)) as f:
			reader = csv.reader(f)
			for row in reader:
				img, label = row
				label = int(label)
				images.append(img)
				labels.append(label)
		assert len(images) == len(labels)
		return images, labels

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		# idx =>0~__len__
		img, label = self.images[idx], self.labels[idx]
		tf = transforms.Compose([
			lambda x: Image.open(x).convert('RGB'),
			transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
			transforms.RandomRotation(15),
			transforms.CenterCrop(self.resize),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
			                     std=[0.229, 0.224, 0.225])
		])
		img = tf(img)
		label = torch.tensor(label)

		return img, label


