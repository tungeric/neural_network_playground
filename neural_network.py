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

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # X = W * I
        # output at node = sigmoid function(x)
        # repeat for hidden and output nodes
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # error = target - output
        output_errors = targets - final_outputs

        # hidden layer error is output_errors split by weights recombined at hidden nodes
        # errors_hidden = weights_hidden (transposed) * errors(output)
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update weights for links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        # update weights for links between input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))

    def query(self, inputs_list):
        # convert inputs list into 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # X = W * I
        # output at node = sigmoid function(x)
        # repeat for hidden and output nodes
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
