import numpy
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork


def main():
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3

    # create neural network object
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # load training data into a list
    training_data_file = open("dataset/mnist_train_100.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # train the neural network!
    for record in training_data_list:
        # split record by commas
        all_values = record.split(',')
        # scale and shift inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create target output values (all 0.01 except desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass

    # test de network
    test_data_file = open("dataset/mnist_test_10.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # scorecard lol
    scorecard = []
    for record in test_data_list:
        # split record by commas
        all_values = record.split(',')
        correct_label = int(all_values[0])
        print(correct_label, "correct label")
        # scale and shift inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = n.query(inputs)
        # index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        print(label, "network's answer")
        # append correct or incorrect to list
        if label == correct_label:
            # network's answer matches!
            scorecard.append(1)
        else:
            # boo
            scorecard.append(0)
    print(scorecard)


if __name__ == "__main__":
    main()
