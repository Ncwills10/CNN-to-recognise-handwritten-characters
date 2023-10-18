import numpy as np

def spacialExtractor(x, numberColumns):
    
    '''
    This takes in a row of the dataframe as a numpy array (which represents 
    one square image) and reinstates the spacial relations by assuming the
    original image was flattened according to a np.flatten. The outpupt is 
    a square 2D numpy array.
    '''
    
    if len(x) % np.sqrt(len(x)) != 0:
        raise Exception("The input length must be a square number.")
    
    output = np.empty((1, numberColumns))
    
    for i in range(numberColumns, len(x) + 1, numberColumns):
        row = x[i - numberColumns : i]
        row = np.reshape(row, (1, numberColumns))
        output = np.concatenate((output, row), axis = 0)
    
    output = output[1:]
    
    if np.shape(output)[0] != np.shape(output)[1]:
        raise Exception("Output is not square")
       
    return output