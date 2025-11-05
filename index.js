import NeuralNetwork from "./nn/nn.js";

let nn = new NeuralNetwork(2, [5, 5, 5], 1)

let X = [[1, 1], [1, 0], [0, 1], [0, 0]]
let y = [[0], [1], [1], [0]]
nn.train(X, y, 10000, 0.05)

console.log(nn.feedforward(X))