# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron in pytorch (using only tensors)
# Written by Mathieu Lefort
#
# Distributed under BSD licence.
# ------------------------------------------------------------------------

import gzip, numpy, torch
    
if __name__ == '__main__':
	batch_size = 5 # number of data read each time
	nb_epochs = 10 # number of time the dataset will be read
	eta = 0.00001 # learning rate
	
	# data loading
	((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))

	# model and weights initialisation
	w = torch.empty((data_train.shape[1],label_train.shape[1]),dtype=torch.float)
	b = torch.empty((1,label_train.shape[1]),dtype=torch.float)
	torch.nn.init.uniform_(w,-0.001,0.001)
	torch.nn.init.uniform_(b,-0.001,0.001)

	nb_data_train = data_train.shape[0]
	nb_data_test = data_test.shape[0]
	indices = numpy.arange(nb_data_train)
	for n in range(nb_epochs):
		# shuffling the (indices of the) data
		numpy.random.shuffle(indices)
		# reading all the training (indices of the) data
		for i in range(0,nb_data_train,batch_size):
			# getting the inputs
			x = data_train[indices[i:i+batch_size]]
			# computing the output of the model
			y = torch.mm(x,w)+b
			# getting the true labels
			t = label_train[indices[i:i+batch_size]]
			# updating weights
			grad = (t-y)
			w += eta * torch.mm(x.T,grad)
			b += eta * grad.sum(axis=0)

		# testing the model (test accuracy is computed during training for monitoring)
		acc = 0.
		# reading all the testing data
		for i in range(nb_data_test):
			# getting the input
			x = data_test[i:i+1]
			# computing the output of the model
			y = torch.mm(x,w)+b
			# getting the true label
			t = label_test[i:i+1]
			# checking if the output is correct
			acc += torch.argmax(y,1) == torch.argmax(t,1)
		# printing the accuracy
		print(acc/nb_data_test)


print(y[0])
print(label_test[0])
print(w.shape)
print(b.shape)
print(batch_size)
print(nb_epochs)
print(nb_data_train)
print(nb_data_test)
print(y)
print(acc)
print(indices.shape)