import os

import numpy as np
import matplotlib.pyplot as plt

from Layers import Helpers, Initializers
from Layers.Conv import Conv
from Layers.BatchNormalization import BatchNormalization
from Layers.Flatten import Flatten
from Layers.FullyConnected import FullyConnected
from Layers.ReLU import ReLU
from Layers.SoftMax import SoftMax
from NeuralNetwork import NeuralNetwork
from Optimization import Constraints, Loss, Optimizers


# Define network parameters
iterations = 200
input_image_shape = (1, 8, 8)
conv_stride_shape = (1, 1)
convolution_shape = (1, 3, 3)
categories = 10
batch_size = 150
num_kernels = 4

# Create optimizer used for gradient descent
adam_with_l2 = Optimizers.Adam(5e-3, 0.98, 0.999)
adam_with_l2.add_regularizer(Constraints.L2_Regularizer(8e-2))

# Create weight and bias initializer
weight_initializer = Initializers.He()
bias_initializer = Initializers.Constant(0.1)

# Create loss function for classification
loss_layer = Loss.CrossEntropyLoss()

# Load UCI ML hand-written digits datasets
data_layer = Helpers.DigitData(batch_size)

# Create Neural Network and append layers
net = NeuralNetwork(adam_with_l2, weight_initializer, bias_initializer)
net.data_layer = data_layer
net.loss_layer = loss_layer

conv1 = Conv(conv_stride_shape, convolution_shape, num_kernels)
batch_norm2 = BatchNormalization(num_kernels)
relu1 = ReLU()
flatten1 = Flatten()
fc1 = FullyConnected(np.prod((num_kernels, *input_image_shape[1:])), categories)
softmax = SoftMax()

net.append_layer(conv1)
net.append_layer(batch_norm2)
net.append_layer(relu1)
net.append_layer(flatten1)
net.append_layer(fc1)
net.append_layer(softmax)

# Start training
net.train(iterations)

# Test accuracy on test data set
data, labels = net.data_layer.get_test_set()
results = net.test(data)
accuracy = Helpers.calculate_accuracy(results, labels)
print('On the UCI ML hand-written digits dataset we achieve an accuracy of: {}%'.format(accuracy * 100.))

# Plot training loss
fig = plt.figure()
plt.plot(net.loss, '-x')
plt.title("Loss function for training a Convnet on the Digit dataset")
fig.savefig(os.path.join("./", "TestConvNet.png"), bbox_inches='tight', pad_inches=0)