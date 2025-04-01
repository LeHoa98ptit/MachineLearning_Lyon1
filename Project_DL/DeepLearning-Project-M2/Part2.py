import torch, gzip
from matplotlib import pyplot as plt
import torch.nn as nn
import numpy as np
import torch.optim as optim

# read data and print size of tensor
((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open("mnist.pkl.gz"))

print("Size of data_train", data_train.shape)
print("Size of lable_train", label_train.shape)
print("Size of data_test", data_test.shape)
print("Size of label_test", label_test.shape)

# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(np.array(data_train[i], dtype='float').reshape(28, 28), cmap=plt.get_cmap('gray'))
# show the figure
plt.show()

# implement
batch_size = 5  # number of data read each time
nb_epochs = 10  # number of time the dataset will be read

# initialising the data loaders
train_dataset = torch.utils.data.TensorDataset(data_train, label_train)
test_dataset = torch.utils.data.TensorDataset(data_test, label_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define simple MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(self.relu(x))
        x = self.fc2(x)
        return x

# Define
input_size = data_train.shape[1]   # amount of features size (corresponding MNIST image size)
hidden_size = 64  # the number of neural in the hidden layer
output_size = label_train.shape[1]  # size of output (corresponding from 0 to 9)

# model initialisation
model = MLP(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss(reduction='sum')
# Using (SGD) và learning rate (η) 0.001
optimizer = optim.SGD(model.parameters(), lr=0.001)

# print model architecture
print(model)

for n in range(nb_epochs):
    # reading all the training data
    for x, t in train_loader:
        # computing the output of the model
        y = model(x)
        # updating weights
        loss = criterion(t, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # testing the model (test accuracy is computed during training for monitoring)
    acc = 0.0
    # reading all the testing data
    for x, t in test_loader:
        # computing the output of the model
        y = model(x)
        # checking if the output is correct
        acc += torch.argmax(y, 1) == torch.argmax(t, 1)

    # printing the accuracy
    print(acc / data_test.shape[0])
