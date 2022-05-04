from Layers.Base import Base

import numpy as np

class Dropout(Base):

    def __init__(self, probability):
        super().__init__()
        self.probability = probability


    def forward(self, input_tensor):
        output = input_tensor

        if not self.testing_phase:
            self.mask = np.random.choice([0, 1], size=input_tensor.shape, p=[1 - self.probability, self.probability]) / self.probability
            output = input_tensor * self.mask

        return output


    def backward(self, error_tensor):
        return error_tensor * self.mask
