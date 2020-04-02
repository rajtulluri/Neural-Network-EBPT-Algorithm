class NeuralNetwork:

    """Class that implements a basic neural network, Delta learning rule with 
    forward computation and backward propagation"""
    
    def __init__(self,layers,alpha):
        
        #Initializing all variables and randomly inititializing the weights
        self.weights=[]
        self.numlayers = len(layers) - 1
        self.learning_rate = alpha
        self.output = []
        self.layer_outputs = []
        self.Error = []
        self.epochs = 0
        
        for k in range(0,self.numlayers):
            #Random weight values between -2 and 2 float type
            #Using Xavier weight initialization heuristic
            xavier_heuristic = (np.sqrt(6)/np.sqrt(layers[k] + layers[k+1]))
            self.weights.append(np.random.uniform(xavier_heuristic,-xavier_heuristic,[layers[k],layers[k+1]]))
            
    def sigmoid(self,dot_prod):
        #Based on the logistic funtion, calculated for unipolar neuron
        return 1/(1 + np.exp(-dot_prod))
    
    def feedforward(self,x):
        
        #Using the weight matrices and Input, we calculate outputs for each layer of the neuron.
        self.layer_outputs = []
        self.layer_outputs.append(np.array(self.sigmoid(np.dot(np.array(x),self.weights[0]))))
        
        for k in range(1,self.numlayers):
            self.layer_outputs.append(self.sigmoid(np.dot(self.layer_outputs[k-1],self.weights[k])))
            
        self.output.append(self.layer_outputs[self.numlayers-1])
        
    def backpropagation(self,x,y):
        
        #Based on the output of the network and target variable, the error is propagated back in the network
        #Delta error is calculated for the output layer (right most layer)
        self.layer_outputs.insert(0,x)
        output_delta_error = (y - self.output[-1]) * (self.output[-1]*(1-self.output[-1]))
        self.delta_error = [output_delta_error.reshape(-1,1)]
        n = self.numlayers
        
        #Delta propagated backwards in the network, calculate delta error for each layer
        for i in range(self.numlayers-1,0,-1):
            self.delta_error.append(np.dot(self.weights[i],self.delta_error[n-i-1]) * (self.layer_outputs[i]*(1 - self.layer_outputs[i])).reshape(-1,1))
            
        #Using Delta error, weight updation
        for i in range(self.numlayers-1,-1,-1):
            self.weights[i] += np.round(self.learning_rate * np.transpose(self.delta_error[n-i-1] * self.layer_outputs[i]),2)
            
        self.layer_outputs.remove(x)
    
    def cumulative_error(self,y):
        #For each pattern (input tuple) submitted, the cummulative error of neurons in output layer calculated
        return np.round((1/2)*sum((y-self.output[-1])**2),2)
    
    def train_network(self,inp,target,thresh=0.3,max_iter=1500):
        
        #Training the network with a train data set and target variables for supervised learning
        E=1
        #The network learns until either cummulative error threshold is crossed or #epochs (max_iter) completed
        while np.round(E,2) > thresh and self.epochs < max_iter:
                        
            if len(inp) != len(target) :
                print("Length mismatch")
                break
                
            E = 0
            
            #Submitting each pattern (input tuple) to the network and updating weights according to the target
            for i in range(0,len(target)):
                self.feedforward(inp[i])
                E += self.cumulative_error(target[i])
                self.backpropagation(inp[i],target[i])
                
            self.epochs += 1
            self.Error.append(E)
            
            print("Epoch: ",self.epochs, 
                  " Learning rate: ",self.learning_rate,
                  " Error: ",E
                 )
        
    def predict(self,inp):
        
        #Based on a trained model, make predictions on given input data set
        self.predicted = []
        
        for x in inp:
            self.feedforward(x)
            self.predicted.append((self.output[-1],np.round(self.output[-1]).astype('int')))
            
        return self.predicted
    
    def accuracy(self,pred,actual):
        
        #Calculate the accuracy of the model, by percentage of correct predictions
        correct = 0
        
        for p,a in zip(pred,actual):
            if np.array_equal(p,a):
                correct+=1
                
        return correct/(len(actual)) * 100
