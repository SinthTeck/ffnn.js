import NeuralNetwork from "./nn/nn.js";

let nn = new NeuralNetwork(2, 4, 1)

let X = [[1, 0], [0, 0], [0, 1], [1, 1]]
let y = [[1], [0], [1], [0]]
nn.train(X, y, 50000, 0.1)

console.log(nn.feedforward(X))