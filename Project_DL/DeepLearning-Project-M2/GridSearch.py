import itertools
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
output_size = label_train.shape[1]  # size of output (corresponding from 0 to 9)

# Define a list of values to search for hidden_size and learning_rate
hidden_sizes = [64, 128, 256]
learning_rates = [0.01, 0.1, 0.001]

best_accuracy = 0.0
best_hidden_size = 0
best_learning_rate = 0.0

# Define loss function and optimizer
criterion = nn.MSELoss(reduction='sum')

for hidden_size, learning_rate in itertools.product(hidden_sizes, learning_rates):
    # Create a new model with the current hidden_size and learning_rate
    model = MLP(input_size, hidden_size, output_size)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(nb_epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluate the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            _, labels = torch.max(target, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    # Check if this combination is the best so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_hidden_size = hidden_size
        best_learning_rate = learning_rate

print(f"Best Hidden Size: {best_hidden_size}, Best Learning Rate: {best_learning_rate}, Best Accuracy: {best_accuracy:.2f}%")
