import numpy
from neural_network import NeuralNetwork


def main():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3

    # create neural network object
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    x = numpy.array([1, 2, 3], ndmin=8) .T


if __name__ == "__main__":
    main()
