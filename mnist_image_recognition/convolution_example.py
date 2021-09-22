import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import tensorflow as tf

# use the gpu if you can, otherwise use the cpu
device = "cuda" if torch.cuda.is_available() else "cpu"


class Network(nn.Module):
	def __init__(self, input_dim, input_channels, output_dim, alpha=0.001):
		super(Network, self).__init__()
		
		# convlution stage, simple as possible
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=(2, 2)),
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
		)
		
		# rather than be complicated and use a function to calculate the output, we can just
		# calculate it by taking the size of an example input
		temp = torch.zeros((1, input_channels, *input_dim), dtype=torch.float)
		size = self.conv(temp).view(-1).shape[0]
		# use this size as input for the first layer of the typical network
		self.model = nn.Sequential(
			nn.Linear(size, 512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, output_dim),
		)
		
		self.opt = optim.Adam(self.parameters(), lr=alpha)
		self.to(device)
	
	def forward(self, x):
		# alternate way of viewing:
		# batch_size = x.shape[0]
		# x = self.conv(x)
		# x = x.view(batch_size, -1)
		# x = self.model(x)
		# return x
		return self.model(self.conv(x).view(x.shape[0], -1))


# load the mnist dataset from tensorflow, because it's super easy that way
# and I like super easy
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

input_dim = 28, 28  # each image is 28px by 28px
input_channels = 1  # the image only has one channel, a 0 to 255 b/w value
output_dim = 10  # there are 10 possible classifications, 0-9
test_size = x_test.shape[0]  # for the model accuracy test

# grab the testing data now so we don't have it in a loop
test_data = torch.tensor(x_test, dtype=torch.float, device=device).reshape(test_size, input_channels, *input_dim)
test_data_target = torch.tensor(y_test, dtype=torch.long, device=device)

model = Network(input_dim=input_dim, input_channels=input_channels, output_dim=output_dim)

# both somewhat arbitrarily chosen.
# a batch size of 10 would make you train on all the examples, but it takes longer and isn't really needed for this
epochs = 100
batch_size = 32

# repeat the whole training process [epoch] times
for e in range(epochs):
	epoch_loss = []
	# loop through each i value skipping by the batch size
	for i in range(0, x_train.shape[0], batch_size):
		# grab the data from the i value to the i value plus batch size and reshape it for the convolution layer
		data = torch.tensor(x_train[i:i + batch_size], dtype=torch.float, device=device).reshape(batch_size,
																								 input_channels,
																								 *input_dim)
		# cross entropy needs these values to be a long
		data_target = torch.tensor(y_train[i:i + batch_size], dtype=torch.long, device=device)
		
		out = model(data)
		
		# cross entropy because multiclassification
		# NOTE: don't use softmax, just straight linear output is fine.
		# softmax won't work
		loss = F.cross_entropy(out, data_target)
		loss_item = loss.cpu().item()
		
		model.opt.zero_grad()  # clear the gradients
		loss.backward()  # add stuff to the gradients
		model.opt.step()  # update the weights with the gradients and the learning rate
		epoch_loss.append(loss_item)
	
	avg_loss = np.mean(epoch_loss)
	print(f"Epoch: {e} - Epoch Loss: {avg_loss}", end="")
	
	# we can grab the data at the top, because the data doesn't change, only the model
	out = model(train_data).max(-1)[1]
	
	# calculate the accuracy, right / total
	acc = sum([1 if out[i] == train_data_target[i] else 0 for i in range(test_size)]) / test_size
	print(f" - Accuracy: {round(acc * 100, 1)}%")

