import { transpose, map, ones, random, dotMultiply, add, multiply, sum, zeros, subtract } from "mathjs";

class NeuralNetwork {
    inputs_size
    hidden1_size
    hidden2_size
    output_size

    constructor(inputs_size, hidden1_size, hidden2_size, output_size) {
        this.inputs_size = inputs_size
        this.hidden1_size = hidden1_size
        this.hidden2_size = hidden2_size
        this.output_size = output_size

        this.weights_input_hidden1 = map(ones(inputs_size, hidden1_size), random)
        this.weights_hidden1_hidden2 = map(ones(hidden1_size, hidden2_size), random)
        this.weights_hidden2_output = map(ones(hidden2_size, output_size), random)

        this.bias_hidden1 = zeros(1, hidden1_size)
        this.bias_hidden2 = zeros(1, hidden2_size)
        this.bias_output = zeros(1, output_size)
    }

    sigmoid(x) {
        return 1/(1+Math.E**(-x));
    }

    sigmoid_derivative(x) {
        return x*(1-x);
    }

    feedforward(X) {
        this.hidden1_activation = add(multiply(X, this.weights_input_hidden1), this.bias_hidden1)
        this.hidden1_output = map(this.hidden1_activation, this.sigmoid)

        this.hidden2_activation = add(multiply(this.hidden1_output, this.weights_hidden1_hidden2), this.bias_hidden2)
        this.hidden2_output = map(this.hidden2_activation, this.sigmoid)

        this.output_activation = add(multiply(this.hidden2_output, this.weights_hidden2_output), this.bias_output)
        this.predicted_output = map(this.output_activation, this.sigmoid)

        return this.predicted_output
    }

    backward(X, y, learningRate = 0.1) {
        let output_error = subtract(y, this.predicted_output);
        
        let output_delta = dotMultiply(output_error, map(this.predicted_output, this.sigmoid_derivative))
        
        let hidden2_error = multiply(output_delta, transpose(this.weights_hidden2_output))
        let hidden2_delta = dotMultiply(hidden2_error, map(this.hidden2_output, this.sigmoid_derivative))
    
        let hidden1_error = multiply(hidden2_delta, transpose(this.weights_hidden1_hidden2))
        let hidden1_delta = dotMultiply(hidden1_error, map(this.hidden1_output, this.sigmoid_derivative))

        this.weights_hidden2_output = add(this.weights_hidden2_output, dotMultiply(multiply(transpose(this.hidden2_output), output_delta), learningRate))
        this.bias_output = sum(output_delta) * learningRate

        this.weights_hidden1_hidden2 = add(this.weights_hidden1_hidden2, dotMultiply(multiply(transpose(this.hidden1_output), hidden2_delta), learningRate))
        this.bias_hidden2 = sum(hidden2_delta) * learningRate

        this.weight_input_hidden1 = add(this.weights_input_hidden1, dotMultiply(multiply(transpose(X), hidden1_delta), learningRate))
        this.bias_hidden1 = sum(hidden1_delta) * learningRate
    }

    train(X, y, epochs, learningRate) {
        for(let i = 0; i<epochs; i++) {
            this.feedforward(X)
            this.backward(X, y, learningRate)
        }
    }
}

export default NeuralNetwork;