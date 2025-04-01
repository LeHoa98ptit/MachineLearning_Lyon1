# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron in pytorch (with pytorch tools)
# Written by Mathieu Lefort
#
# Distributed under BSD licence.
# ------------------------------------------------------------------------

import gzip,numpy,torch

import gzip, numpy, torch
    
if __name__ == '__main__':
	batch_size = 5  # number of data read each time
	nb_epochs = 10  # number of time the dataset will be read
	eta = 0.00001  # learning rate
	
	# data loading
	((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))
	# initialising the data loaders
	train_dataset = torch.utils.data.TensorDataset(data_train,label_train)
	test_dataset = torch.utils.data.TensorDataset(data_test,label_test)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

	# initialising the model and weights
	model = torch.nn.Linear(data_train.shape[1],label_train.shape[1])
	torch.nn.init.uniform_(model.weight,-0.001,0.001)
	# initialising the optimiser
	loss_func = torch.nn.MSELoss(reduction='sum')
	optim = torch.optim.SGD(model.parameters(), lr=eta)

	for n in range(nb_epochs):
		# reading all the training data
		for x,t in train_loader:
			# computing the output of the model
			y = model(x)
			# updating weights
			loss = loss_func(t,y)
			loss.backward()
			optim.step()
			optim.zero_grad()
			
		# testing the model (test accuracy is computed during training for monitoring)
		acc = 0.
		# reading all the testing data
		for x,t in test_loader:
			# computing the output of the model
			y = model(x)
			# checking if the output is correct
			acc += torch.argmax(y,1) == torch.argmax(t,1)
		# printing the accuracy
		print(acc/data_test.shape[0])
