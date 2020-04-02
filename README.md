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
