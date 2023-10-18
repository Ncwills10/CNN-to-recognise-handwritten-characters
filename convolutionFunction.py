import numpy as np
from spacialExtractor import spacialExtractor

def crossCorrelation(input, lossGradient, filterSize):
    
    '''
    This function performs a cross-correlation of input with lossGradient. The
    stride is 1 and both input and lossGradient are 2D square numpy arrays. 
    filterSize is a number representing the size of the square filter for the 
    relevant layer. The loss gradient is padded uniformly by 
    padding = (filterSize - 1) / 2, as these layers represent the gradient 
    w.r.t the original padding, which is not needed. Therefore, we remove the 
    outer padding layers, before performing the cross-correlation. The output
    is the loss gradient w.r.t the filters of the relevant layer.
    '''
    
    inputHeight, inputWidth = np.shape(input)
    lossGradientHeight, lossGradientWidth = np.shape(lossGradient)
    
    if inputHeight != inputWidth:
        raise Exception("The input must be square.")
    if lossGradientHeight != lossGradientWidth:
        raise Exception("The loss gradient must be square")
    if filterSize % 2 == 0:
        raise Exception("The filter size must be odd.")
    
    padding = int((filterSize - 1) / 2)
    
    lossGradient = lossGradient[padding : lossGradientHeight - padding, padding : lossGradientWidth - padding]
    
    lossGradientHeight, lossGradientWidth = np.shape(lossGradient)
    
    outputHeight, outputWidth = inputHeight - lossGradientHeight + 1, inputWidth - lossGradientWidth + 1
        
    flatOutput = np.array([])
    for i in range(outputHeight):
        for j in range(outputWidth):
            overlap = input[i : i + lossGradientHeight, j : j + lossGradientWidth]
            elementWiseMultiplication = overlap * lossGradient
            sum = np.sum(elementWiseMultiplication)    
            flatOutput = np.append(flatOutput, sum)
            
    dFilters = spacialExtractor(flatOutput, outputWidth)
    
    return dFilters

def fullConvolve(filter, lossGradient):
    
    '''
    This function carries out a full convolution of the relevant filter for the
    layer with the loss gradient. Both are square 2D numpy arrays. The filter
    is rotated 180 degrees and padded. The loss gradient is already padded 
    uniformly by oldPadding = (filterSize - 1) / 2, as these layers represent 
    the gradient w.r.t the original padding, which is not needed. However, the 
    crossCorrelation function defined above removes this for us, so no action
    is needed here. Note the difference between convolution and 
    cross-correlation. The output is the loss gradient w.r.t the input to the
    relevant layer. It is also a square 2D numpy array. The stride is 1.
    '''
    
    filterHeight, filterWidth = np.shape(filter)
    lossGradientHeight, lossGradientWidth = np.shape(lossGradient)
    
    if filterHeight != filterWidth or lossGradientHeight != lossGradientWidth:
        raise Exception("The filter and the loss gradient must both be square.")
    if filterHeight % 2 == 0:
        raise Exception("The filter size must be odd.")
    
    filter = np.rot90(filter, 2)
    
    oldPadding = (filterHeight - 1) / 2
    
    lossGradientHeight = lossGradientHeight - 2 * oldPadding
    
    padding = int(lossGradientHeight - 1)
    
    filter = np.pad(filter, padding, mode = 'constant')
    
    dInput = crossCorrelation(filter, lossGradient, filterHeight)
    
    return dInput
    
def crossCorrelationRetainDimensions(image, filter):
    
    '''
    This function takes in a square 2D numpy array image and cross-correlates
    it with a square 2D numpy array filter, outputting another square 2D numpy
    array of the same dimensions as the image. The output is the output of the
    relevant convolutional layer. The stride is 1.
    '''
    
    imageHeight, imageWidth = np.shape(image)
    filterHeight, filterWidth = np.shape(filter)
    
    if imageHeight != imageWidth or filterHeight != filterWidth:
        raise Exception("The image and the filter must both be square.")
    if filterHeight % 2 == 0:
        raise Exception("The filter size must be odd.")
    
    outputHeight, outputWidth = imageHeight, imageWidth
    
    padding = int((filterHeight - 1) / 2)
    
    image = np.pad(image, padding, mode = 'constant')
        
    flatOutput = np.array([])
    for i in range(outputHeight):
        for j in range(outputWidth):
            overlap = image[i : i + filterHeight, j : j + filterWidth]
            elementWiseMultiplication = overlap * filter
            sum = np.sum(elementWiseMultiplication)    
            flatOutput = np.append(flatOutput, sum)
            
    output = spacialExtractor(flatOutput, outputWidth)
    
    return output