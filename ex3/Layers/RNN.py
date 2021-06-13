from Layers.Base import Base
from Layers.Sigmoid import Sigmoid
from Layers.TanH import TanH
from Layers.FullyConnected import FullyConnected
import numpy as np

class RNN(Base):

    def __init__(self, input_size, hidden_size, output_size):
        # super().__init__() # Causes error because of initialization of standard weights in base-class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Internal layers
        self.fc_hidden = FullyConnected(input_size + hidden_size, hidden_size)
        self.tanh = TanH()
        self.fc_output = FullyConnected(hidden_size, output_size)
        self.sigmoid = Sigmoid()

        # Inputs and activations for different timesteps
        self.sigmoid_activations = []
        self.fc_hidden_inputs = []
        self.tanh_activations = []

        self.mem = False
        self.trainable = True
        self.hidden_state = np.zeros((1, hidden_size))

    
    @property
    def memorize(self):
        return self.mem

    @memorize.setter
    def memorize(self, value):
        self.mem = value

    @property
    def gradient_weights(self):
        return self.fc_hidden.gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self.fc_hidden.gradient_weights = value

    @property
    def optimizer(self):
        return self.fc_hidden.optimizer

    @optimizer.setter
    def optimizer(self, value):
        self.fc_hidden.optimizer = value


    @property
    def weights(self):
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, value):
        self.fc_hidden.weights = value


    def forward(self, input_tensor):
        
        if not self.mem:
            self.hidden_state = np.zeros((1, self.hidden_size))
            self.fc_hidden_inputs.clear()
            self.sigmoid_activations.clear()
            self.tanh_activations.clear()

        output_tensor = np.empty((len(input_tensor), self.output_size))

        for i, t_input in enumerate(input_tensor):
            concat = np.concatenate((t_input[np.newaxis, :], self.hidden_state), axis=1)
            fc_hidden_output = self.fc_hidden.forward(concat)
            self.hidden_state = self.tanh.forward(fc_hidden_output)
            fc_output_output = self.fc_output.forward(self.hidden_state)
            output = self.sigmoid.forward(fc_output_output)

            self.fc_hidden_inputs.append(self.fc_hidden.input_tensor)
            self.tanh_activations.append(self.hidden_state)
            self.sigmoid_activations.append(output)

            output_tensor[i] = output.squeeze()

        return output_tensor



    def backward(self, error_tensor):
        next_error_tensor = np.zeros((len(error_tensor), self.input_size))
        fc_output_gradient_weights = np.zeros_like(self.fc_output.weights)
        fc_hidden_gradient_weights = np.zeros_like(self.fc_hidden.weights)

        gradient_hidden_state = np.zeros_like(self.hidden_state)
        
        for i, sample in enumerate(error_tensor[::-1]):
            # Set input_tensors for gradient calculation
            self.sigmoid.activation = self.sigmoid_activations[-1 - i]
            self.fc_output.input_tensor = np.concatenate((self.tanh_activations[-1 - i], [[1]]), axis=1)
            self.tanh.activation = self.tanh_activations[-1 - i]
            self.fc_hidden.input_tensor = self.fc_hidden_inputs[-1 - i]

            # Backpropagate
            error_sigmoid = self.sigmoid.backward(sample[np.newaxis, :])
            error_fc_output = self.fc_output.backward(error_sigmoid)
            error_tanh = self.tanh.backward(error_fc_output + gradient_hidden_state)
            error_fc_hidden = self.fc_hidden.backward(error_tanh)

            next_error_tensor[-1 - i] = error_fc_hidden[:, :self.input_size]
            gradient_hidden_state = error_fc_hidden[:, self.input_size:]

            # Save gradients w.r.t. the weights
            fc_output_gradient_weights += self.fc_output.gradient_weights
            fc_hidden_gradient_weights += self.fc_hidden.gradient_weights
        
        self.gradient_weights = fc_hidden_gradient_weights

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.fc_output.weights = self.optimizer.calculate_update(self.fc_output.weights, fc_output_gradient_weights)


        return next_error_tensor


    def calculate_regularization_loss(self):
        # sum up the regularization loss from the optimizers/layers incorporated in the RNN cell
        ...


    def initialize(self, weights_initializer, bias_initializer):
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)
