import numpy as np

class Optimizer:

    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):

    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        if self.regularizer is not None:
            updated_weights -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        return updated_weights


class SgdWithMomentum(Optimizer):

    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        updated_weights = weight_tensor + self.v
        if self.regularizer is not None:
            updated_weights -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        return updated_weights


class Adam(Optimizer):

    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.iter = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor ** 2
        v_corr = self.v / (1 - self.mu ** self.iter)
        r_corr = self.r / (1 - self.rho ** self.iter)
        self.iter += 1
        updated_weights = weight_tensor - self.learning_rate * v_corr / (np.sqrt(r_corr) + np.finfo(float).eps)
        if self.regularizer is not None:
            updated_weights -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
            
        return updated_weights