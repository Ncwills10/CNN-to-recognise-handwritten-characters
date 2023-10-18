import numpy as np
from convolutionFunction import crossCorrelation, fullConvolve, crossCorrelationRetainDimensions
from activationFunction import softmax

class convolutionLayer:
    
    '''
    This class creates objects that carry out the forward and backward 
    propagation of a convolutional layer of a CNN.
    '''    
   
    def forward(self, input, filters, biases):
        
        '''
        input is a 4D numpy array 
        (no. in batch, height, width, no. of channels). filters is a 3D numpy
        array (height, width, no. of filters). biases is a 1D numpy array 
        (no. of filters). This carries out the forward propagation of a 
        convolutional layer and outputs the cross-correlation of the input
        with the filters, with the biases added. The output retains the same 
        no. in batch, height and width as the input, with 
        no. of channels = no. of input channels * no. of filters.
        '''
        
        if np.shape(input)[1] != np.shape(input)[2] or np.shape(filters)[0] != np.shape(filters)[1]:
            raise Exception("The feature maps and the filters must both be square.")
        if len(np.shape(input)) != 4:
            raise Exception("The input should be a 4D array.")
        if len(np.shape(filters)) != 3:
            raise Exception("The filters array should be 3D.")
        if len(np.shape(biases)) != 1:
            raise Exception("The biases should be a 1D array.")
        if len(biases) != np.shape(filters)[2]:
            raise Exception("The biases array should have length equal to no. of filters.")
        
        outputShape = np.shape(input)[1]
        
        output = np.empty((1, outputShape, outputShape, np.shape(input)[3] * np.shape(filters)[2]))
        
        for i in input:
            
            featureMapArray = np.empty((outputShape, outputShape, 1))      
            
            for j in range(np.shape(i)[2]):
                
                featureMap = i[:, :, j]
                
                for k in range(np.shape(filters)[2]):
                
                    bias = biases[k]
                    biasArray = np.full((outputShape, outputShape), bias)   
                    crossCorrelation = crossCorrelationRetainDimensions(featureMap, filters[:, :, k])
                    crossCorrelation = crossCorrelation + biasArray
                    crossCorrelation = np.reshape(crossCorrelation, (outputShape, outputShape, 1))
                    featureMapArray = np.concatenate((featureMapArray, crossCorrelation), axis = 2)
              
            featureMapArray = featureMapArray[:, :, 1:]
            featureMapArray = np.reshape(featureMapArray, (1, outputShape, outputShape, np.shape(featureMapArray)[2]))
            
            output = np.concatenate((output, featureMapArray), axis = 0)       
        
        output = output[1:]
        
        return output
    
    def backward(self, input, filters, dz):
        
        '''
        input is the input to the layer and is a 4D numpy array 
        (no. in batch, height, width, no. of channels). filters is a 3D 
        numpy array (height, width, no. of filters). dz is the loss gradient 
        w.r.t the output of the layer and is a 4D numpy array
        (no. in batch, height, width, no. of channels). This carries out the 
        backward propagation of a convoluational layer, returning the loss
        gradient w.r.t the input, the filters and the biases. 
        All the gradients have the same shape as the corresponding array
        (e.g. dFilters has same shape as filters).
        dInput = full convolution of the filters with dz. 
        dFilters = cross-correlation of the input with dz.
        dBiases = average of dz along each dimension except for depth. This is
        because the same bias was added for each filter. It is a 1D array 
        (no. of filters).
        '''
        
        if len(np.shape(input)) != 4:
            raise Exception("The input should be a 4D array.")
        if len(np.shape(filters)) != 3:
            raise Exception("The filters array should be 3D.")
        if len(np.shape(dz)) != 4:
            raise Exception("dz should be a 4D array.")  
        if np.shape(dz)[3] != np.shape(input)[3] * np.shape(filters)[2]:
            raise Exception("The no. of channels of dz should be equal to the no. of input channels multiplied by the number of filters.")
        
        filterNumber = np.shape(filters)[2]
        padding = int((np.shape(filters)[1] - 1) / 2)
        
        dFilterHeight = np.shape(input)[1] - np.shape(dz)[1] + 2 * padding + 1
        dFilterWidth = np.shape(input)[2] - np.shape(dz)[2] + 2 * padding + 1
        dInputHeight = np.shape(filters)[0] + np.shape(dz)[1] - 2 * padding - 1
        dInputWidth = np.shape(filters)[1] + np.shape(dz)[2] - 2 * padding - 1
        
        # This finds the error w.r.t the filters.
        
        dFilters = np.empty((1, dFilterHeight, dFilterWidth, filterNumber))
        
        for i in range(np.shape(input)[0]):

            featureMapArray = input[i, :, :, :]
            
            allChannelErrors = np.empty((1, dFilterHeight, dFilterWidth, filterNumber))
            
            for j in range(np.shape(featureMapArray)[2]):
                
                featureMap = featureMapArray[:, :, j]
                k = (j + 1) * filterNumber
                relevantError = dz[i, :, :, k - filterNumber : k]
                oneChannelError = np.empty((dFilterHeight, dFilterWidth, 1))
                
                for l in range(filterNumber):

                    correlation = crossCorrelation(featureMap, relevantError[:, :, l], np.shape(filters)[0])
                    correlation = np.reshape(correlation, (dFilterHeight, dFilterWidth, 1))
                    oneChannelError = np.concatenate((oneChannelError, correlation), axis = 2)
                    
                oneChannelError = oneChannelError[:, :, 1:]
                oneChannelError = np.reshape(oneChannelError, (1, dFilterHeight, dFilterWidth, filterNumber))
                allChannelErrors = np.concatenate((allChannelErrors, oneChannelError), axis = 0)
                
            allChannelErrors = allChannelErrors[1:]
            
            averagedChannelError = np.sum(allChannelErrors, axis = 0) / np.shape(allChannelErrors)[0]
            averagedChannelError = np.reshape(averagedChannelError, (1, dFilterHeight, dFilterWidth, filterNumber))
            dFilters = np.concatenate((dFilters, averagedChannelError), axis = 0)
        
        dFilters = dFilters[1:]
        dFilters = np.sum(dFilters, axis = 0) / np.shape(dFilters)[0]
        
        # Now to find the error w.r.t the input
        
        dInput = np.empty((1, dInputHeight, dInputWidth, np.shape(input)[3]))
        dBiases = np.empty((1, np.shape(dz)[1], np.shape(dz)[2], filterNumber))
        
        for i in range(np.shape(dz)[0]):
            
            errorForOneInBatch = np.empty((1, dInputHeight, dInputWidth, np.shape(input)[3]))
            biasErrorForOneInBatch = np.empty((np.shape(dz)[1], np.shape(dz)[2], np.shape(input)[3], 1))
            
            for j in range(filterNumber):
                
                filter = filters[:, :, j]
                relevantErrorArray = dz[i, :, :, j : np.shape(dz)[3] : filterNumber]
                biasFromOneFilterArray = dz[i, :, :, j : np.shape(dz)[3] : filterNumber]
                biasFromOneFilterArray = np.reshape(biasFromOneFilterArray, (np.shape(dz)[1], np.shape(dz)[2], np.shape(input)[3], 1))
                biasErrorForOneInBatch = np.concatenate((biasErrorForOneInBatch, biasFromOneFilterArray), axis = 3)
                
                errorsFromOneFilter = np.empty((dInputHeight, dInputWidth, 1))
                
                for k in range(np.shape(relevantErrorArray)[2]):
                    
                    relevantError = relevantErrorArray[:, :, k]
                    fullConvolution = fullConvolve(filter, relevantError)
                    fullConvolution = np.reshape(fullConvolution, (dInputHeight, dInputWidth, 1))
                    errorsFromOneFilter = np.concatenate((errorsFromOneFilter, fullConvolution), axis = 2)
                    
                errorsFromOneFilter = errorsFromOneFilter[:, :, 1:]
                errorsFromOneFilter = np.reshape(errorsFromOneFilter, (1, dInputHeight, dInputWidth, np.shape(relevantErrorArray)[2]))
                errorForOneInBatch = np.concatenate((errorForOneInBatch, errorsFromOneFilter), axis = 0)
            
            errorsFromOneFilter = errorForOneInBatch[1:]
            averagedErrorForOneInBatch = np.sum(errorForOneInBatch, axis = 0) / np.shape(errorForOneInBatch)[0]
            averagedErrorForOneInBatch = np.reshape(averagedErrorForOneInBatch, (1, dInputHeight, dInputWidth, np.shape(input)[3]))
            
            biasErrorForOneInBatch = biasErrorForOneInBatch[:, :, :, 1:]
            averagedBiasErrorForOneInBatch = np.sum(biasErrorForOneInBatch, axis = 2) / np.shape(biasErrorForOneInBatch)[2]
            averagedBiasErrorForOneInBatch = np.reshape(averagedBiasErrorForOneInBatch, (1, np.shape(dz)[1], np.shape(dz)[2], filterNumber))
            dBiases = np.concatenate((dBiases, averagedBiasErrorForOneInBatch), axis = 0)
            
            dInput = np.concatenate((dInput, averagedErrorForOneInBatch), axis = 0)
        
        dInput = dInput[1:]
        dBiases = dBiases[1:]
        
        rowAverage = np.sum(dBiases, axis = 1) / np.shape(dBiases)[1]
        
        dBiases = np.sum(rowAverage, axis = 1) / np.shape(rowAverage)[1]
        
        dBiases = np.sum(dBiases, axis = 0) / np.shape(dBiases)[0]
        
        return dInput, dFilters, dBiases

class maxPoolingLayer:
    
    '''
    This class creates objects that carry out the forward and backward
    propagation of a max pooling layer.
    '''
    
    def forward(self, input, gridSize):
                
        '''
        This carries out the forward propagation. It takes a 4D numpy array
        input (no. in batch, height, width, no. of channels), along with 
        gridSize, the length of the square grid used to pool. The output 
        retains the same (no. in batch, no. of channels) dimensions, with 
        an output height and output width = input height (or input width) - 
        gridSize + 1.
        '''
        
        if len(np.shape(input)) != 4:
            raise Exception("The input should be a 4D array.")
        if np.shape(input)[1] != np.shape(input)[2]:
            raise Exception("The input height and width should be the same.")
        if gridSize > np.shape(input)[1]:
            raise Exception("gridSize must be smaller than or equal to the input height/width.")
        
        outputHeight = np.shape(input)[1] - gridSize + 1
        outputWidth = np.shape(input)[2] - gridSize + 1
        
        output = np.empty((1, outputHeight, outputWidth, np.shape(input)[3]))
        
        for i in input:
            
            outputVolume = np.empty((outputHeight, outputWidth, 1))
            
            for j in range(np.shape(i)[2]):
                
                featureMap = i[:, :, j]
                height = featureMap[0]
                width = featureMap[1]
                    
                outputGrid = np.empty((1, outputWidth))
                
                for k in range(outputHeight):
                    
                    row = np.array([])
                    
                    for l in range(outputWidth):
                        
                        relevantArray = featureMap[k : k + gridSize, l : l + gridSize]
                        max = np.max(relevantArray)
                        row = np.append(row, max)
                        
                    row = np.reshape(row, (1, len(row)))
                    outputGrid = np.concatenate((outputGrid, row), axis = 0)
                    
                outputGrid = outputGrid[1:]
                
                outputGrid = np.reshape(outputGrid, (outputHeight, outputWidth, 1))
                outputVolume = np.concatenate((outputVolume, outputGrid), axis = 2)
                
            outputVolume = outputVolume[:, :, 1:]
            
            outputVolume = np.reshape(outputVolume, (1, outputHeight, outputWidth, np.shape(input)[3]))
            output = np.concatenate((output, outputVolume), axis = 0)
            
        output = output[1:]
        
        return output
    
    def backward(self, input, dz):
        
        '''
        This carries out the back propagation. It takes the input to the layer 
        (no. in batch, height, width, no. of channels) and the loss gradient
        w.r.t the output of the layer. The output is the same size as the
        input to the layer. allMaxIndicesArray is a 3D array 
        (no. in batch x no. of kernels x 5). The 5 columns are as follows: row
        no. of max value in input, column no. of max value in input, row no. of
        corresponding dz value, column no. of corresponding dz value, channel 
        no. for that max value.
        '''
        
        output = np.zeros((np.shape(input)[0], np.shape(input)[1], np.shape(input)[2], np.shape(input)[3]))
        gridSize = np.shape(input)[1] - np.shape(dz)[1] + 1
        
        indicesArray = np.empty((1, np.shape(dz)[1] * np.shape(dz)[2] * np.shape(dz)[3], 5))
        
        for i in range(np.shape(input)[0]):
            
            indices = np.empty((1, 5))
            
            for j in range(np.shape(input)[3]):
                
                featureMap = input[i, :, :, j]
                height = featureMap[0]
                width = featureMap[1]
                actualMaxIndicesArray = np.empty((1, 4))
                
                for k in range(np.shape(dz)[1]):
                    
                    for l in range(np.shape(dz)[2]):
                        
                        relevantArray = featureMap[k : k + gridSize, l : l + gridSize]
                        max = np.max(relevantArray)
                        
                        maxIndices = np.array([])
                        
                        for m in np.where(relevantArray == max):
                    
                            maxIndices = np.append(maxIndices, m[0])
                            
                        indicesTopLeftOfKernel = np.array([k, l])
                        actualMaxIndices = indicesTopLeftOfKernel + maxIndices
                        actualMaxIndices = np.concatenate((actualMaxIndices, indicesTopLeftOfKernel), axis = 0)
                        actualMaxIndices = np.reshape(actualMaxIndices, (1, 4))
                        actualMaxIndicesArray = np.concatenate((actualMaxIndicesArray, actualMaxIndices), axis = 0)
                        
                actualMaxIndicesArray = actualMaxIndicesArray[1:]
                channelColumn = np.full((np.shape(actualMaxIndicesArray)[0], 1), j)
                actualMaxIndicesArray = np.concatenate((actualMaxIndicesArray, channelColumn), axis = 1)
                indices = np.concatenate((indices, actualMaxIndicesArray), axis = 0)
        
            indices = indices[1:]
            indices = np.reshape(indices, (1, np.shape(dz)[1] * np.shape(dz)[2] * np.shape(dz)[3], 5))
            indicesArray = np.concatenate((indicesArray, indices), axis = 0)
            
        indicesArray = indicesArray[1:]
        
        for n in range(np.shape(indicesArray)[0]):
            
            for o in indicesArray[n, :, :]:
                
                output[n, int(o[0]), int(o[1]), int(o[4])] = dz[n, int(o[2]), int(o[3]), int(o[4])]
        
        return output               
                
class flatteningLayer:
    
    '''
    This class creates objects that carry out the forward and backward
    propagation of a flattening layer.
    '''
    
    def __init__(self, shape = np.array([])):
        
        self.shape = shape
    
    def forward(self, input):
        
        '''
        This carries out the forward propagation. It takes a 4D numpy array
        input (no. in batch, height, width, no. of channels) and outputs a 2D
        numpy array (no. in batch, volume).
        '''
        
        self.shape = np.shape(input)
        flattenedArray = np.empty((1, np.shape(input)[1] * np.shape(input)[2] * np.shape(input)[3]))
                
        for i in input:
            
            flat = i.flatten()
            flat = np.reshape(i, (1, len(flat)))
            flattenedArray = np.concatenate((flattenedArray, flat), axis = 0)
        
        flattenedArray = flattenedArray[1:]
        
        return flattenedArray
              
    def backward(self, dz):
        
        '''
        This carries out backward propagation. The input is the loss gradient
        w.r.t the output of the layer, which is a 2D numpy array 
        (no. in batch, volume). The output is also dz but reshaped to match
        the input to the layer.
        '''
        
        output = np.empty((1, self.shape[1], self.shape[2], self.shape[3]))
        
        for i in dz:
            
            reshaped = i.reshape(self.shape[1:])
            reshaped = np.reshape(reshaped, (1, self.shape[1], self.shape[2], self.shape[3]))
            output = np.concatenate((output, reshaped), axis = 0)
        
        output = output[1:]           
            
        return output    
        
class fullyConnectedLayer:
    
    '''
    This class creates objects that carry out the forward and backward 
    propagation of a fully connected layer.
    '''
    
    def forward(self, input, numberNeurons, weights, biases):
        
        '''
        This carries out the forward propagation. It takes in a 2D numpy array
        input (no. in batch, volume), as well as the number of neurons for the
        layer, the 2D numpy array weights (volume, no. of neurons) and a 1D 
        numpy array biases (no. of neurons). The output is a 2D numpy array 
        (no. in batch, no. of neurons).
        '''
        
        biases = np.reshape(biases, (1, numberNeurons))
        
        biasesArray = np.empty((1, numberNeurons))
        
        for i in range(np.shape(input)[0]):
            
            biasesArray = np.concatenate((biasesArray, biases), axis = 0)
            
        biasesArray = biasesArray[1:]
        
        dotProduct = np.dot(input, weights)
        output = dotProduct + biases
        
        return output

    def backward(self, input, weights, dz):
        
        '''
        This performs backward propagation. It takes in the 2D numpy array
        input to the layer (no. in batch, volume), as well as the 2D numpy
        array weights for the layer (volume, no. neurons) and the 2D numpy 
        array dz (no. in batch, no. of neurons). It outputs the loss gradient
        w.r.t the input, the weights and the biases. Each loss gradient has the
        same shape as the variable it is respect to.
        '''
    
        dInput = np.dot(dz, np.transpose(weights))
        dWeights = np.dot(np.transpose(input), dz)
        dBiases = np.sum(dz, axis = 0) / np.shape(dz)[0]
        
        return dInput, dWeights, dBiases
    
class outputLayer:
    
    '''
    This class creates objects that carry out the forward and backward 
    propagation for an output layer.
    '''
    
    def forward(self, input, weights, biases):
        
        '''
        This carries out the forward propagation. It takes a 2D numpy array
        input (no. in batch, length of previous layer), the 2D numpy array 
        weights (length of previous layer, no. of classes) and the 1D numpy 
        array biases (no. of classes). The output is a 2D numpy
        array (no. in batch, no. of classes) containing the probabilities for
        each class.
        '''
        
        biases = np.reshape(biases, (1, np.shape(weights)[1]))
        
        biasesArray = np.empty((1, np.shape(weights)[1]))
        
        for i in range(np.shape(input)[0]):
            
            biasesArray = np.concatenate((biasesArray, biases), axis = 0)
            
        biasesArray = biasesArray[1:]
        
        dotProduct = np.dot(input, weights)
        unactivatedOutput = dotProduct + biases
        output = softmax(unactivatedOutput)
        
        return output
    
    def backward(self, input, classLabels, weights, probs):
        
        '''
        This carries out the backward propagation. It takes in the 2D numpy 
        array input to the layer (no. in batch, length of previous layer), as
        well as the 1D numpy array class labels (no. in batch), the 2D numpy
        array weights (length of previous layer, no. of classes) and the 2D
        numpy array of probabilities for each class 
        (no. in batch, no. of classes). It outputs the loss gradient w.r.t the
        input to the layer, the weights and the biases. Each loss gradient has 
        the same shape as the variable it is respect to.
        '''
        
        dOutput = probs
        
        for i in range(np.shape(probs)[0]):
            
            dOutput[i, int(classLabels[i])] -= 1
             
        dBiases = np.sum(dOutput, axis = 0) / np.shape(dOutput)[0]
        dInput = np.dot(dOutput, np.transpose(weights))
        dWeights = np.dot(np.transpose(input), dOutput)
        
        return dInput, dWeights, dBiases