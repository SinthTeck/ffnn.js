import { transpose, map, ones, random, dotMultiply, add, multiply, sum, zeros, subtract } from "mathjs";

class NeuralNetwork {
    inputs_size
    hidden_size
    output_size

    constructor(inputs_size, hidden_size, output_size) {
        this.inputs_size = inputs_size
        this.hidden_sizes = hidden_size
        this.output_size = output_size
        this.weights = []
        this.bias = []

        this.activations = []
        this.outputs = []

        this.output = 0
        
        this.weights.push(map(ones(inputs_size, hidden_size[0]), random))
        for(let i=0; i<this.hidden_sizes.length; i++) {
            if(i===this.hidden_sizes.length-1){
                break;
            }
            this.weights.push(map(ones(this.hidden_sizes[i], this.hidden_sizes[i+1]), random))
        }
        this.weights.push(map(ones(this.hidden_sizes[this.hidden_sizes.length-1], output_size), random))

        for(let i=0; i<this.hidden_sizes.length; i++){
            this.bias.push(zeros(1, this.hidden_sizes[i]))
        }

        this.bias.push(zeros(1, output_size))
    }

    sigmoid(x) {
        return 1/(1+Math.E**(-x));
    }

    sigmoid_derivative(x) {
        return x*(1-x);
    }

    feedforward(X) {
        this.activations = []
        this.outputs = []
        for(let i=0; i<this.weights.length; i++) {
            if(i===0){
                this.activations.push(add(multiply(X, this.weights[i]), this.bias[i]))
                this.outputs.push(map(this.activations[i], this.sigmoid))
            } else {
                this.activations.push(add(multiply(this.outputs[i-1], this.weights[i]), this.bias[i]))
                this.outputs.push(map(this.activations[i], this.sigmoid))
            }
        }
        this.output = this.outputs[this.outputs.length-1]

        return this.output
    }

    backward(X, y, learningRate = 0.1) {
        const deltas = [];
        const numLayers = this.weights.length;
        
        const output_error = subtract(y, this.output);
        const output_delta = dotMultiply(output_error, map(this.output, this.sigmoid_derivative));
        deltas.push(output_delta);
        
        for(let i = numLayers - 2; i >= 0; i--) {
            const error = multiply(deltas[0], transpose(this.weights[i + 1]));
            const delta = dotMultiply(error, map(this.outputs[i], this.sigmoid_derivative));
            deltas.unshift(delta);
        }
        
        for(let i = 0; i < numLayers; i++) {
            let input;
            if(i === 0) {
                input = X;
            } else {
                input = this.outputs[i - 1];
            }
            
            const weight_gradient = multiply(transpose(input), deltas[i]);
            this.weights[i] = add(this.weights[i], dotMultiply(weight_gradient, learningRate));
            
            const bias_gradient = sum(deltas[i], 0);
            this.bias[i] = add(this.bias[i], dotMultiply(bias_gradient, learningRate));
        }
    }

    train(X, y, epochs, learningRate) {
        for(let i = 0; i<epochs; i++) {
            this.feedforward(X)
            this.backward(X, y, learningRate)
        }
    }
}

export default NeuralNetwork;