
import numpy as np
from loguru import logger

class RegressionNeuralNetwork:
    
    def __init__(self, input_size : int = 1, h1_neurons : int = 16, h2_neurons : int = 4, output_size : int = 1, lambda_error : float = 0.1) -> None:
        
        self.input_size     = input_size
        self.h1_neurons     = h1_neurons
        self.h2_neurons     = h2_neurons
        self.output_size    = output_size
        
        self.x  = None
        self.y  = None
        self.w1 = None
        self.w2 = None
        self.w3 = None
        
        self.accuracy : list[float] = list()
        self.loss     : list[float] = list()
        
        # regularization
        self.lambda_error = lambda_error
               
        
    def init(self, x : np.ndarray, y : np.ndarray) -> None:
        
        self.x          = x
        self.y          = y
        self.loss       = list()
        self.accuracy   = list()
        
        self._init_weights()
    
    
    def train(self, epochs : int = 10) -> None:
        
        self._train_network(epochs)
                
    
    def predict(self, input : np.ndarray) -> np.ndarray:
        
        output = list()
        
        for i in range(len(input)):
            
            x = input[i].reshape(1, self.input_size)
            y = self._forward_propogate(x)
            y = y[0]
            
            output.append(y)
            
        return np.array(output).reshape(len(input), self.output_size)
    
    def _init_weights(self) -> None: 
        
        self.w1 = np.random.normal(size = (self.input_size, self.h1_neurons))
        self.w2 = np.random.normal(size = (self.h1_neurons, self.h2_neurons))
        self.w3 = np.random.normal(size = (self.h2_neurons, self.output_size))
            
    def _forward_propogate(self, x : np.ndarray) -> np.ndarray:
        
        input = x
        
        # hidden layer-1
        h1 = input.dot(self.w1)
        h1 = self.sigmoid(h1)
        
        # hidden layer-2 
        h2 = h1.dot(self.w2)
        h2 = self.relu(h2)
        
        # output layer
        output = h2.dot(self.w3)
        
        # pass the output without any transformation
        return output
    
    
    def _backward_propogate(self, x : np.ndarray, y : np.ndarray) -> None:
        
        input = x
        
        # hidden layer-1
        z1 = input.dot(self.w1)
        a1 = self.sigmoid(z1)
        
        # hidden layer-2 
        z2 = a1.dot(self.w2)
        a2 = self.relu(z2)
        
        # output 
        z3      = a2.dot(self.w3)
        output  = z3
        
        z3_error = output - y
        
        a2_der      = self.relu_derivative(a2)
        #logger.info(f'{self.w3.shape}  {z3_error.shape}, {a2_der.shape}')
        z2_error    = np.multiply(self.w3.dot(z3_error.T).T, a2_der)
        
        a1_der      = self.sigmoid_derivative(a1)
        #logger.info(f'{self.w2.shape}  {z2_error.shape}, {a1_der.shape}')
        z1_error    = np.multiply(self.w2.dot(z2_error.T).T, a1_der)
        
        w1_adj = input.T.dot(z1_error)
        w2_adj = a1.T.dot(z2_error)
        w3_adj = a2.T.dot(z3_error)
        
        self.w1 -= self.lambda_error * w1_adj
        self.w2 -= self.lambda_error * w2_adj
        self.w3 -= self.lambda_error * w3_adj
        
    
    def _train_network(self, epochs : int = 10) -> None:
        
        logger.info(f'training network for epochs: {epochs}')
        
        x = self.x
        y = self.y
        
        train_n = len(x)
        
        for i in range(epochs):
            
            epoch_loss = list()
            
            for j in range(len(self.x)):
                
                x_i = x[i].reshape(1, self.input_size)
                y_i = y[i].reshape(1, self.output_size)
                
                output = self._forward_propogate(x_i)
                
                loss = self._calculate_MSE(y_i, output)
                
                self._backward_propogate(x_i, y_i)
                
                epoch_loss.append(loss)
                
            current_loss = sum(epoch_loss) / train_n
            
            self.loss.append(current_loss)
            
            logger.info(f'Epoch-{i + 1}: MSE- {current_loss}')
            
        logger.info(f'completed network training')
          
        
    def _calculate_MSE(self, y : np.ndarray, output : np.ndarray) -> float:
        
        mse = np.sum(np.square(output - y)) / len(y)
        
        return mse
            
    def sigmoid(self, x : np.ndarray) -> np.ndarray:
        
        return 1. / (1. + np.exp(-x))
    
    def sigmoid_derivative(self, a : np.ndarray) -> np.ndarray:
        
        return np.multiply(a, 1 - a)
    
    def relu(self, x : np.ndarray) -> np.ndarray:
        
        x = x.copy()
        
        x[x <= 0.] = 0.
        
        return x
    
    def relu_derivative(self, a : np.ndarray) -> np.ndarray:
        
        a = a.copy()
        
        a[a <= 0.] = 0.
        a[a > 0.]  = 1.
        
        return a
        
        
    
    
    
        
        
