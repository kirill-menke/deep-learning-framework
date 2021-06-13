import numpy as np

class Base:
    
    def __init__(self):
        self.trainable = False
        self.testing_phase = False
        self.weights = np.zeros(0)