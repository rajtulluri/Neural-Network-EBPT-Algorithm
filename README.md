# Neural-Network-EBPT-Algorithm

Neural Networks consist of the following components

   An input layer, x <br>
   An arbitrary amount of hidden layers <br>
   An output layer, ŷ <br>
   A set of weights between each layer, W <br>
   A choice of activation function for each hidden layer, σ. In this tutorial, we’ll use a Sigmoid activation function. <br>

The output ŷ of a simple 2-layer Neural Network is:
   
   ŷ = sigmoid(W<sub>2</sub> * sigmoid(W<sub>1</sub> * X))
   
You might notice that in the equation above, the weights W is the only variable that affects the output ŷ.
Naturally, the right values for the weights determines the strength of the predictions. 
The process of fine-tuning the weightsfrom the input data is known as training the Neural Network.

Each iteration of the training process consists of the following steps:

   Calculating the predicted output ŷ, known as feedforward
   Updating the weights and biases, known as backpropagation

For every feedforward performed, we calculate the cumulative error as: 
   
   E = (1/2) * sum((y - ŷ)<sup>2</sup>)
   
 
