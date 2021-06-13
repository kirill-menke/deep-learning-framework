from Layers.Base import Base
import numpy as np

class Sigmoid(Base):

    def __init__(self):
        super().__init__()
        self.cnt_f = 0
        self.cnt_b = 0

    def forward(self, input_tensor):
        self.activation = 1 / (1 + np.exp(-input_tensor))
        return self.activation

    def backward(self, error_tensor):
        return error_tensor * self.activation * (1 - self.activation)