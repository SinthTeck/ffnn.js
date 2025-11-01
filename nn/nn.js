import { transpose, map, ones, random, dotMultiply, add, multiply, sum, zeros, subtract } from "mathjs";

class NeuralNetwork {
    inputs_size
    hidden_size
    output_size

    constructor(inputs_size, hidden_size, output_size) {
        this.inputs_size = inputs_size
        this.hidden_size = hidden_size
        this.output_size = output_size

        this.weight_input_hidden = map(ones(inputs_size, hidden_size), random)
        this.weights_hidden_output = map(ones(hidden_size, output_size), random)

        this.bias_hidden = zeros(1, hidden_size)
        this.bias_output = zeros(1, output_size)
    }

    sigmoid(x) {
        return 1/(1+Math.E**(-x));
    }

    sigmoid_derivative(x) {
        return x*(1-x);
    }

    feedforward(X) {
        this.hidden_activation = add(multiply(X, this.weight_input_hidden), this.bias_hidden)
        this.hidden_output = map(this.hidden_activation, this.sigmoid)

        this.output_activation = add(multiply(this.hidden_output, this.weights_hidden_output), this.bias_output)
        this.predicted_output = map(this.output_activation, this.sigmoid)

        return this.predicted_output
    }

    backward(X, y, learningRate = 0.1) {
        let output_error = subtract(y, this.predicted_output);
        
        let output_delta = dotMultiply(output_error, map(this.predicted_output, this.sigmoid_derivative))
        
        let hidden_error = multiply(output_delta, transpose(this.weights_hidden_output))
        let hidden_delta = dotMultiply(hidden_error, map(this.hidden_output, this.sigmoid_derivative))

        this.weights_hidden_output = add(this.weights_hidden_output, dotMultiply(multiply(transpose(this.hidden_output), output_delta), learningRate))
        this.bias_output = sum(output_delta) * learningRate

        this.weight_input_hidden = add(this.weight_input_hidden, dotMultiply(multiply(transpose(X), hidden_delta), learningRate))
        this.bias_output = sum(hidden_delta) * learningRate 
    }

    train(X, y, epochs, learningRate) {
        for(let i = 0; i<epochs; i++) {
            this.feedforward(X)
            this.backward(X, y, learningRate)
        }
    }
}

export default NeuralNetwork;