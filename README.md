# Neural-Network-EBPT-Algorithm

Neural Networks consist of the following components

   An input layer, x <br>
   An arbitrary amount of hidden layers <br>
   An output layer, ŷ <br>
   A set of weights between each layer, W <br>
   A choice of activation function for each hidden layer, σ. In this tutorial, we’ll use a Sigmoid activation function. <br>

![Architecture](https://github.com/rajtulluri/Neural-Network-EBPT-Algorithm/blob/master/Misc/architec.png)

The output ŷ of a simple 2-layer Neural Network is:
   
   ŷ = sigmoid(W<sub>2</sub> * sigmoid(W<sub>1</sub> * X))
   
You might notice that in the equation above, the weights W is the only variable that affects the output ŷ.
Naturally, the right values for the weights determines the strength of the predictions. 
The process of fine-tuning the weightsfrom the input data is known as training the Neural Network.

Each iteration of the training process consists of the following steps:

   Calculating the predicted output ŷ, known as feedforward
   Updating the weights and biases, known as backpropagation
   
![pipeline of training](https://github.com/rajtulluri/Neural-Network-EBPT-Algorithm/blob/master/Misc/pipeline.png)

For every feedforward performed, we calculate the cumulative error as: 
   
   E = (1/2) * sum((y - ŷ)<sup>2</sup>)
   
This will be our loss function.
Our goal in training is to find the best set of weights that minimizes the loss function.

In order to know the appropriate amount to adjust the weights by, we need to know the derivative of the loss function with respect to the weights.

![Loss graph](https://github.com/rajtulluri/Neural-Network-EBPT-Algorithm/blob/master/Misc/Loss.png)

If we have the derivative, we can simply update the weights by increasing/reducing with it(refer to the diagram above). This is known as gradient descent. For the derivative:

![Loss derivative](https://github.com/rajtulluri/Neural-Network-EBPT-Algorithm/blob/master/Misc/LossDerivative.png)

Using this derivative, we can update the weights.



