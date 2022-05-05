import copy

class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None


    @property
    def phase(self):
        return self.layers[0].testing_phase

    @phase.setter
    def phase(self, value):
        for layer in self.layers:
            layer.testing_phase = value


    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        regularizer_loss = 0
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            if self.optimizer.regularizer is not None:
                regularizer_loss += self.optimizer.regularizer.norm(layer.weights)
        return self.loss_layer.forward(input_tensor, self.label_tensor) + regularizer_loss


    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)


    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)


    def train(self, iterations):
        self.phase = False
        for _ in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    
    def test(self, input_tensor):
        self.phase = True
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
    
