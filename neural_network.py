import numpy
import scipy.special

class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate

        # link weight matrices
        # TODO: make nodes its own object
        # option 1: random numbers within -0.5 - 0.5
        # self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        # self.who = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        # option 2: use 1/ sqrt(# of incoming links) (in this case 3x3)
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self):
        pass

    def query(self, inputs_list):
        # convert inputs list into 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # X = W * I
        # output at node = sigmoid function(x)
        # repeat for hidden and output nodes
        hidden_inputs = numpy.dot(self.wih, inputs_list)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
