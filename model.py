import numpy as np
import pandas as pd
from spacialExtractor import spacialExtractor
from layer import convolutionLayer, maxPoolingLayer, flatteningLayer, fullyConnectedLayer, outputLayer
from activationFunction import ReLU4D, ReLU2D
from PIL import Image


class model:
    
    def __init__(self, batchNumber, classesNumber, imageSize, filterNumbers, filterSize, neuronNumber, poolingSize):
        
        '''
        This class creates objects that represent a model of a CNN. It takes in
        the number of images per batch, the number of classes, the size of the 
        square input images, a 1D numpy array where each element is the number 
        of filters in each layer, the size of the square filters, the number of
        neurons in the one fully connected layer and the size of the pooling 
        grid.
        '''
        
        self.batchNumber = batchNumber
        self.classesNumber = classesNumber
        self.imageSize = imageSize
        self.filterNumbers = filterNumbers
        self.filterSize = filterSize
        self.neuronNumber = neuronNumber
        self.poolingSize = poolingSize   
        
        self.filters = np.array([])
        self.filterBiases = np.array([])
        self.layerBiases = np.array([])
        self.outputWeights = np.array([])
        self.outputBiases = np.array([])
        
        self.shapeList = np.array([])
         
    def calculateLoss(self, probs, classLabels):
        
        '''
        This method returns the categorical cross entropy loss for a multiclass
        classification. It takes in a 2D numpy array input probs 
        (no. in batch, no. of classes) and the 1D numpy array class labels 
        (no. in batch). The output is the same shape as the probabilities,
        with every element equal to 0 except for the element corresponding to
        the class label.
        '''
                
        lossArray = np.zeros((np.shape(probs)[0], np.shape(probs)[1]))
        
        for i in range(np.shape(probs)[0]):

            lossArray[i, int(classLabels[i])] = -1 * np.log(probs[i, int(classLabels[i])])
        
        return lossArray
                    
    def train(self, data, numberEpochs, alpha):
        
        '''
        This method trains the model. It takes in data, a pandas data frame
        with the rows representing different images and the columns 
        representing different pixels of a greyscale image. The first column 
        contains the class labels. It also takes in the number of epochs and 
        alpha, the learning rate.
        '''
        
        layerConvolution = convolutionLayer()
        layerPooling = maxPoolingLayer()
        layerFlattening = flatteningLayer()
        layerConnected = fullyConnectedLayer()
        layerOutput = outputLayer()
        
        activation2D = ReLU2D()
        activation4D = ReLU4D()
        
        for i in range(numberEpochs):
            
            print("Epoch no: %d" % i)
        
            for j in range(0, data.shape[0], self.batchNumber):
            
                if j + self.batchNumber <= data.shape[0]:
        
                    batch = data.iloc[j : j + self.batchNumber]  
        
                else:
        
                    break
        
                numberInBatch = len(batch)
                classLabels = batch.iloc[:, 0]
                classLabels = np.array(classLabels)
                batch = batch.iloc[:, 1:]
                batch = np.array(batch)
            
                batch = np.reshape(batch, (numberInBatch, self.imageSize, self.imageSize, 1))
            
            # Now we carry out forward propagation. 
        
            # Initialising the filter weights and biases.
            
                if i == 0 and j == 0:
                    
                    self.filters = np.random.randn(self.filterSize, self.filterSize, np.sum(self.filterNumbers))
                    self.filterBiases = np.random.randn(np.sum(self.filterNumbers))
                       
            # First, the convolution layers.

                sum = self.filterNumbers[0]
                
                inputsConv = np.empty(len(self.filterNumbers), dtype = object)
                inputsActivation = np.empty(len(self.filterNumbers), dtype = object)
                inputsPooled = np.empty(len(self.filterNumbers), dtype = object)
                
                input = batch
                
                for k in range(len(self.filterNumbers)):
                    
                    inputsConv[k] = input
                    filters = self.filters[:, :, sum - self.filterNumbers[k] : sum]
                    biases = self.filterBiases[sum - self.filterNumbers[k] : sum]

                    outputConv = layerConvolution.forward(input, filters, biases)
                    
                    inputsActivation[k] = outputConv

                    outputReLU4D = activation4D.forward(outputConv)
                    
                    inputsPooled[k] = outputReLU4D

                    outputPooled = layerPooling.forward(outputReLU4D, self.poolingSize)
                    
                    if k < len(self.filterNumbers) - 1:
                        sum += self.filterNumbers[k + 1]
                    
                    input = outputPooled

            # Now flattening layer.
               
                outputFlattening = layerFlattening.forward(outputPooled)                
    
            # Now we have one fully connected layer.
            
            # Initialising the layer weights and biases.
            
                if i == 0 and j == 0:
                    
                    self.layerWeights = np.random.randn(np.shape(outputFlattening)[1], self.neuronNumber)
                    self.layerBiases = np.random.randn(self.neuronNumber)
            
                outputConnected = layerConnected.forward(outputFlattening, self.neuronNumber, self.layerWeights, self.layerBiases)
                
                outputReLU2D = activation2D.forward(outputConnected)
        
            # Finally, we have the output layer.
            
            # Initialising the output weights and biases.
            
                if i == 0 and j == 0:
                    
                    self.outputWeights = np.random.randn(self.neuronNumber, self.classesNumber)
                    self.outputBiases = np.random.randn(self.classesNumber)
            
                probs = layerOutput.forward(outputReLU2D, self.outputWeights, self.outputBiases)
                
            # Now we calculate the loss using the method defined above.
        
                lossArray = self.calculateLoss(probs, classLabels)
            
            # Now to perform back propagation.
        
            # First, I back propagate through the output layer.
        
                dOutputReLU2D, dWeights, dBiases = layerOutput.backward(outputReLU2D, classLabels, self.outputWeights, probs)
                self.outputWeights = self.outputWeights - alpha * dWeights
                self.outputBiases = self.outputBiases - alpha * dBiases
                
            
            # Now I back propagate through the fully connected layer.    
                
                dOutputConnected = activation2D.backward(outputConnected, dOutputReLU2D)
            
                dOutputFlattening, dWeights, dBiases = layerConnected.backward(outputFlattening, self.layerWeights, dOutputConnected)
                self.layerWeights = self.layerWeights - alpha * dWeights
                self.layerBiases = self.layerBiases - alpha * dBiases
        
            # Now I back propagate through the flattening layer.
        
                dOutputPooled = layerFlattening.backward(dOutputFlattening)
            
            # Now I back propagate through the convolution layers.
            
                dInput = dOutputPooled

                sum = np.sum(self.filterNumbers)
                
                for k in range(len(self.filterNumbers) - 1, -1, -1):
                    
                    filters = self.filters[:, :, sum - self.filterNumbers[k] : sum]
                    
                    dOutputReLU4D = layerPooling.backward(inputsPooled[k], dInput)
                    
                    dOutputConv = activation4D.backward(inputsActivation[k], dOutputReLU4D)
                    
                    dInput, dFilters, dBiases = layerConvolution.backward(inputsConv[k], filters, dOutputConv)
                    
                    self.filters[:, :, sum - self.filterNumbers[k] : sum] = self.filters[:, :, sum - self.filterNumbers[k] : sum] - alpha * dFilters
                    
                    self.filterBiases[sum - self.filterNumbers[k] : sum] = self.filterBiases[sum - self.filterNumbers[k] : sum] - alpha * dBiases
                    
                    sum -= self.filterNumbers[k]
        
        print("End of training")
            
    def predict(self, image, classIndexToAscii):
            
        '''
        This method takes in a greyscale image as a 2D numpy array 
        (height, width) and predicts what letter is shown using the trained
        weights and biases. It shows the image and prints the prediction.
        It also takes in classIndexToAscii, a 2D numpy array where the 
        first column is the class label and the seconde column is the Ascii
        value.
        '''

        layerConvolution = convolutionLayer()
        layerPooling = maxPoolingLayer()
        layerFlattening = flatteningLayer()
        layerConnected = fullyConnectedLayer()
        layerOutput = outputLayer()
        
        activation2D = ReLU2D()
        activation4D = ReLU4D()
        
        image = np.rot90(image, 3)
        outputImage = Image.fromarray(image)
        outputImage.show()
        
        image = np.reshape(image, (1, np.shape(image)[0], np.shape(image)[1], 1))
        sum = self.filterNumbers[0]
        input = image
        
        for k in range(len(self.filterNumbers)):
           
            filters = self.filters[:, :, sum - self.filterNumbers[k] : sum]
            biases = self.filterBiases[sum - self.filterNumbers[k] : sum]
            
            outputConv = layerConvolution.forward(input, filters, biases)
            outputReLU4D = activation4D.forward(outputConv)
            outputPooled = layerPooling.forward(outputReLU4D, self.poolingSize)
                
            if k < len(self.filterNumbers) - 1:
                sum += self.filterNumbers[k + 1]
                
            input = outputPooled

        outputFlattening = layerFlattening.forward(outputPooled)
        outputConnected = layerConnected.forward(outputFlattening, self.neuronNumber, self.layerWeights, self.layerBiases)
        outputReLU2D = activation2D.forward(outputConnected)
        probs = layerOutput.forward(outputReLU2D, self.outputWeights, self.outputBiases)

        probs = np.reshape(probs, np.shape(probs)[1])
        
        predictedIndex = np.where(probs == np.max(probs))[0][0]
        asciiValues = classIndexToAscii[:, 1]
        predictedLetterAscii = asciiValues[predictedIndex]
        predictedLetter = chr(predictedLetterAscii)
        
        print(predictedLetter)
    
    def accuracy(self, data):
        
        '''
        This method runs all the testing data and determines the accuracy of 
        the model. data is a pandas data frame with the rows representing
        different images and the columns representing different pixels of a 
        greyscale image. The first column contains the class labels.
        ''' 
        
        layerConvolution = convolutionLayer()
        layerPooling = maxPoolingLayer()
        layerFlattening = flatteningLayer()
        layerConnected = fullyConnectedLayer()
        layerOutput = outputLayer()
        
        activation2D = ReLU2D()
        activation4D = ReLU4D()
        
        numberImages = len(data)
        classLabels = data.iloc[:, 0]
        classLabels = np.array(classLabels)
        data = data.iloc[:, 1:]
        data = np.array(data)
        data = np.reshape(data, (numberImages, self.imageSize, self.imageSize, 1))       
        
        sum = self.filterNumbers[0]
        
        input = data
        
        for k in range(len(self.filterNumbers)):
           
            filters = self.filters[:, :, sum - self.filterNumbers[k] : sum]
            biases = self.filterBiases[sum - self.filterNumbers[k] : sum]
            
            outputConv = layerConvolution.forward(input, filters, biases)
            outputReLU4D = activation4D.forward(outputConv)
            outputPooled = layerPooling.forward(outputReLU4D, self.poolingSize)
                
            if k < len(self.filterNumbers) - 1:
                sum += self.filterNumbers[k + 1]
                
            input = outputPooled

        outputFlattening = layerFlattening.forward(outputPooled)
        outputConnected = layerConnected.forward(outputFlattening, self.neuronNumber, self.layerWeights, self.layerBiases)
        outputReLU2D = activation2D.forward(outputConnected)
        probs = layerOutput.forward(outputReLU2D, self.outputWeights, self.outputBiases)

        counter = 0
        
        for i in range(np.shape(probs)[0]):
            
            predictedIndex = np.where(probs == np.max(probs))[0][0]
            actualIndex = classLabels[i]  
            if predictedIndex == actualIndex:
                counter += 1
        
        print(counter * 100 / numberImages)
        
    def save(self):
        
        '''
        This method saves all the weights and biases for each layer in a
        separate csv file. The csv files exist already and are overwritten
        everytime the method is ran.
        '''
        
        filters = self.filters
        filterBiases = self.filterBiases
        layerWeights = self.layerWeights
        layerBiases = self.layerBiases
        outputWeights = self.outputWeights
        outputBiases = self.outputBiases
        
        parametersList = [filters, filterBiases, layerWeights, layerBiases, outputWeights, outputBiases]
        shapeList = np.empty(len(parametersList), dtype = object)
        pathsList = ["filters.csv", "filterBiases.csv", "layerWeights.csv", "layerBiases.csv", "outputWeights.csv", "outputBiases.csv"] 
        
        for i in range(len(parametersList)):
            
            shapeList[i] = np.shape(parametersList[i])
            parametersList[i] = parametersList[i].flatten()
            parametersList[i] = pd.DataFrame(parametersList[i])     
            parametersList[i].to_csv(pathsList[i], index = False)
        
        self.shapeList = shapeList
    
    def returnAllParameters(self):
        
        '''
        This method reads the csv files of parameters, manipulates the data
        and returns a list of the parameters.
        '''
        
        filters = np.array([])
        filterBiases = np.array([])
        layerWeights = np.array([])
        layerBiases = np.array([])
        outputWeights = np.array([])
        outputBiases = np.array([])
        
        parametersList = [filters, filterBiases, layerWeights, layerBiases, outputWeights, outputBiases]
        pathsList = ["filters.csv", "filterBiases.csv", "layerWeights.csv", "layerBiases.csv", "outputWeights.csv", "outputBiases.csv"]
        
        for i in range(len(parametersList)):
            
            data = pd.read_csv(pathsList[i])
            data = np.array(data)
            data = np.reshape(data, self.shapeList[i])
            parametersList[i] = data
        
        return parametersList