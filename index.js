import NeuralNetwork from "./nn/nn.js";

let nn = new NeuralNetwork(3, 6, 2, 1)

let X = [[1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
let y = [[1], [0], [0], [0], [0]]
nn.train(X, y, 100000, 0.1)

console.log(nn.feedforward(X))