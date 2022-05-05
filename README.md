# Python CNN Framework

## Description
This is a simple CNN framework based on Python and pure Numpy. 

It can be used for educational purposes (e.g. to understand the inner workings of single layers on a lower level) and for experimentation with simple CNN architectures.

## Features
- ***Layers***
  - FullyConnected
  - Convolution
  - MaxPooling
  - Dropout
  - BatchNormalization
  - Flatten
  - Simple RNN

- ***Activations***
  - Sigmoid
  - ReLU
  - TanH
  - Softmax
  
- ***Optimizers***
  - SGD
  - SGDWithMomentum
  - Adam

- ***Initializers***
  - Constant
  - UniformRandom
  - Xavier/ Glorot
  - He

- ***Loss Functions***
  - CrossEntropyLoss


## Example: Classification on UCI ML hand-written digits dataset
The example code below can be found in `example.py` and demonstrates a simple classification on the UCI ML hand-written digits dataset which was loaded using scikit_learn. The architecture below can be extended and individual layers replaced as required. The available components are located in the folders `./Layers` and `./Optimization`. 
```python
import numpy as np

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
batch_size = 150
num_kernels = 4
categories = 10

input_image_shape = (1, 8, 8)
conv_stride_shape = (1, 1)
convolution_shape = (1, 3, 3)

# Define optimizer used for gradient descent during backpropagation
adam_with_l2 = Optimizers.Adam(5e-3, 0.98, 0.999)
adam_with_l2.add_regularizer(Constraints.L2_Regularizer(8e-2))

# Define weight and bias initializer
weight_initializer = Initializers.He()
bias_initializer = Initializers.Constant(0.1)

# Define loss function for classification
loss_layer = Loss.CrossEntropyLoss()

# Load UCI ML hand-written digits datasets
data_layer = Helpers.DigitData(batch_size)

# Create neural network and its layers
net = NeuralNetwork(adam_with_l2, weight_initializer, bias_initializer)
net.data_layer = data_layer
net.loss_layer = loss_layer

conv1 = Conv(conv_stride_shape, convolution_shape, num_kernels)
batch_norm2 = BatchNormalization(num_kernels)
relu1 = ReLU()
flatten1 = Flatten()
fc1 = FullyConnected(np.prod((num_kernels, *input_image_shape[1:])), categories)
softmax = SoftMax()

# Append layers
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
```
