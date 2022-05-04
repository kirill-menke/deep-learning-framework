import copy
import numpy as np

from Layers.Base import Base
from . import Helpers

class BatchNormalization(Base):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gradient_weights = None
        self.gradient_bias = None
        self.moving_avg = np.array([0])
        self.moving_std = np.array([0])
        self.alpha = 0.8
        self.initialize(channels)
        self._optimizer_weights = None
        self._optimizer_bias = None


    @property
    def optimizer(self):
        return self._optimizer_weights, self._optimizer_bias


    @optimizer.setter
    def optimizer(self, value):
        self._optimizer_weights = value
        self._optimizer_bias = copy.deepcopy(value)


    def initialize(self, channels):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
    

    def forward(self, input_tensor):
        
        self.input_tensor = input_tensor
        input_tensor = self.reformat(input_tensor)

        if self.testing_phase:
            self.norm_tensor = (input_tensor - self.moving_avg) / np.sqrt(self.moving_std**2 + np.finfo(float).eps)
        else:
            self.mu_b = np.mean(input_tensor, axis=0)
            self.std_b = np.std(input_tensor, axis=0)
            self.moving_avg = self.mu_b if not self.moving_avg.all() else self.alpha * self.moving_avg + (1 - self.alpha) * self.mu_b
            self.moving_std = self.std_b if not self.moving_std.all() else self.alpha * self.moving_std + (1 - self.alpha) * self.std_b
            self.norm_tensor = (input_tensor - self.mu_b) / np.sqrt(self.std_b**2 + np.finfo(float).eps)

        return self.reformat(self.weights * self.norm_tensor + self.bias)


    def backward(self, error_tensor):

        error_tensor = self.reformat(error_tensor)
        next_error_tensor = Helpers.compute_bn_gradients(error_tensor, self.reformat(self.input_tensor), self.weights, self.mu_b, self.std_b**2)

        self.gradient_weights = np.sum(error_tensor * self.norm_tensor, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)

        if all(self.optimizer):
            opt_w, opt_b = self.optimizer
            self.weights = opt_w.calculate_update(self.weights, self.gradient_weights)
            self.bias = opt_b.calculate_update(self.bias, self.gradient_bias)

        return self.reformat(next_error_tensor)


    def reformat(self, tensor):
        if tensor.ndim == 4:
            reshape = tensor.reshape(len(tensor), self.channels, -1)                # (B x H x M x N) -> (B x H x M*N)
            transpose = reshape.swapaxes(1, 2)                                      # (B x H x M*N)   -> (B x M*N x H)
            return transpose.reshape(-1, self.channels)                             # (B x M*N x H)   -> (B*M*N x H)
        elif tensor.ndim == 2:
            reshape = tensor.reshape(self.input_tensor.shape[0], -1, self.channels)
            transpose = reshape.swapaxes(1, 2)
            return transpose.reshape(self.input_tensor.shape)