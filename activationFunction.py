import numpy as np
from spacialExtractor import spacialExtractor

class ReLU2D:

    '''
    This class creates objects that carry out the ReLU activation function in 
    the forward and backward propagations on 2D numpy array inputs.
    '''    
    
    def forward(self, input):
        
        '''
        The forward propagation takes an input as defined above. The output is 
        the same shape as the input.
        '''
        
        if len(np.shape(input)) != 2:
            raise Exception("This method requires a 2D input")      
    
        return np.maximum(0, input)  

    def backward(self, input, dz):
        
        '''
        The backward propagation takes the input to the ReLU activation 
        function, as well as the loss gradient w.r.t the output of the 
        function. The input and dz have the same shape. The output is the loss 
        gradient w.r.t the output of the previous layer and has the same shape
        as the input.
        '''
    
        if len(np.shape(input)) != 2:
            raise Exception("The inputs should both be 2D arrays.")
        
        output = np.empty((1, np.shape(input)[1]))
        
        for i in range(np.shape(input)[0]):
            dummyArray = np.array([])
            for j in range(np.shape(input)[1]):
                if input[i, j] <= 0:
                    dummyArray = np.append(dummyArray, 0)
                if input[i, j] > 0:
                    dummyArray = np.append(dummyArray, dz[i, j])
            dummyArray = np.reshape(dummyArray, (1, len(dummyArray)))
            output = np.concatenate((output, dummyArray), axis = 0)
        
        output = output[1:]
        
        return output

class ReLU4D:
    
    '''
    This class creates objects that carry out the forward and backward 
    propagation of the ReLU activation function for 4D array inputs
    (no. in batch, height, width, no. of channels).
    '''
    
    def forward(self, input):
        
        '''
        This method takes an input as defined above. The output is the same 
        shape as the input.
        '''
        
        if len(np.shape(input)) != 4:
            raise Exception("The input should be a 4D array.")

        return np.maximum(0, input)        
    
    def backward(self, input, dz):
        
        '''
        This method takes the input to the function and the loss gradient w.r.t
        the output. Both inputs are the same shape. The output is also the same
        shape and is the loss gradient w.r.t the output of the previous layer.
        '''
        
        if np.shape(input) != np.shape(dz):
            raise Exception("The input and dz should have the same shape.")
        
        output = np.empty((1, np.shape(input)[1], np.shape(input)[2], np.shape(input)[3]))
        
        for i in range(np.shape(input)[0]):
            
            oneOfBatch = np.empty((np.shape(input)[1], np.shape(input)[2], 1))
            
            for j in range(np.shape(input)[3]):
                
                array2D = np.empty((1, np.shape(input)[2]))
                
                for k in range(np.shape(input)[1]):
                    
                    row = np.array([])
                    
                    for l in range(np.shape(input)[2]):
                                           
                        if input[i, k, l, j] <= 0:
                            
                            row = np.append(row, 0)
                            
                        if input[i, k, l, j] > 0:
                            row = np.append(row, dz[i, k, l, j])
    
                    row = np.reshape(row, (1, len(row)))
                    array2D = np.concatenate((array2D, row), axis = 0)
                    
                array2D = array2D[1:]
                array2D = np.reshape(array2D, (np.shape(input)[1], np.shape(input)[2], 1))
                
                oneOfBatch = np.concatenate((oneOfBatch, array2D), axis = 2)
                
            oneOfBatch = oneOfBatch[:, :, 1:]
            oneOfBatch = np.reshape(oneOfBatch, (1, np.shape(input)[1], np.shape(input)[2], np.shape(input)[3]))
            
            output = np.concatenate((output, oneOfBatch), axis = 0)
            
        output = output[1:]
        
        return output            
                
def softmax(input):
    
    '''
    This carries out the softmax activation function on a 2D numpy array input
    (no. in batch, no. of classes). The output array is the same shape as the 
    input and contains the probabilities for each class for each image in the
    batch.
    '''
    
    probs = np.empty((1, np.shape(input)[1]))
    
    for i in input:
        
        exp = np.exp(i - np.max(i))
        sum = np.sum(exp)
        rowProbs = exp / sum
        rowProbs = np.reshape(rowProbs, (1, len(rowProbs)))
        probs = np.concatenate((probs, rowProbs), axis = 0)
    
    probs = probs[1:]
    
    return probs