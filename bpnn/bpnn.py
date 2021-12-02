import numpy as np
from math import exp

class BPNN:

    learning_rate = 0.1
    x_inputs = np.array([[1, 1]])
    weights_hidden = np.array([
        [0.5, 0.4],
        [0.9, 1.0]
    ])
    weights_output = np.array([[-1.2, 1.1]])
    thresholds_hidden = np.array([0.8, -0.1])
    biaseds_hidden = np.array([-1, -1])
    thresholds_output = np.array([0.3])
    biaseds_output = np.array([-1])
    y_desired = np.array([0])

    def __init__(self, _input, y):
        self.x_inputs = _input
        self.y_desired = y

    def update_weights(self, ws_hidden):
        self.weights_hidden = ws_hidden
        return self.weights_hidden

    def weight_updation(self, ws_hidden, weights_output, delta_hideen, delta_outputs, thresholds_hidden, thresholds_output):
        self.weights_hidden

    def activation_function_hidden(x_inputs, weights, thresholds):
        # TODO fix docstring
        """
            Calculate actual output at hidden neurons.
            In this case, neuron 3 and neuron 4

            The actual output at hidden neuron will be input for output layer
            This function return Numpy array if possible. 

            Formula tu calculate actual output, y
            y = sigmoid(sum(xi * wij) - thres)

            where:
                xi  = input variables
                wij = weight values; 
                    i = neuron at input layer, 
                    j = neuron at hidden layer 
        """
        actual_output = []
        x_inputs_hidden = []

        # Multiply all x_input with
        for x in range(len(weights)):
            for x_input in x_inputs:
                actual_output_input_x_hidden = x_input * weights[x]
                actual_output.append(actual_output_input_x_hidden)

        actual_outputs_sum = np.sum(
            np.array(actual_output),
            axis=1
        )

        # Multiply the value
        for x, actual_output_sum in enumerate(actual_outputs_sum):
            # print(actual_output_sum)
            actual_outputs_sum[x] = actual_output_sum - (thresholds[x])

        # Sigmoid activation
        for actual_output_sum in actual_outputs_sum:
            x_inputs_hidden.append(
                round(1 / (1 + exp(-1 * actual_output_sum)), 4))

        return np.array([x_inputs_hidden])

    def calc_err_gradient(y_outputs, errors, weights=[]):
        err_gradient = np.array([])
        if not len(weights) == 0:
            for weight in weights:
                for error in errors:
                    for y_output in (y_outputs):
                        for x in range(len(y_output)):
                            err_gradient = np.append(err_gradient, values=round(
                                y_output[x] * (1 - y_output[x]) * error * weight[x], 4))
        else:
            for error in errors:
                for y_output in y_outputs:
                    for x in range(len(y_output)):
                        err_gradient = np.append(err_gradient, values=round(
                            y_output[x] * (1 - y_output[x]) * error[x], 4))

        return err_gradient

    def calc_delta_weight(x_inputs, err_gradients, biaseds):
        delta_weight = np.array([])
        delta_threshold = np.array([])
        temp = []
        neurons = []

        for x in range(len(err_gradients)):
            temp.clear()
            for x_input in x_inputs:
                for y in range(len(x_input)):
                    temp.append(
                        round(learning_rate * x_input[y] * err_gradients[x], 4))
                neurons.append(temp.copy())

        delta_weight = np.array(neurons)
        print(delta_weight)

        for x, biased in enumerate(biaseds):
            delta_threshold = np.append(delta_threshold, round(
                learning_rate * biased * err_gradients[x], 4))

        return {
            'delta_weight': delta_weight,
            'delta_threshold': delta_threshold
        }
