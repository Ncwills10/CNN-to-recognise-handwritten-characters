import numpy as np
import pandas as pd
from model import model
from spacialExtractor import spacialExtractor
from PIL import Image

trainData = pd.read_csv(r'C:\Users\ncwil\OneDrive\Coding\AI\Hand writing recognition neural network\emnist-balanced-train.csv')
trainData = trainData.iloc[:1000]

cnn = model(1, 47, 28, np.array([32, 64]), 3, 64, 3)          
            
cnn.train(trainData, 1, 0.01)    

testData = pd.read_csv(r'C:\Users\ncwil\OneDrive\Coding\AI\Hand writing recognition neural network\emnist-balanced-test.csv')
testData = testData.iloc[:1000]

cnn.accuracy(testData)
cnn.save()

# Extracting a test image.

'''
oneImage = testData.iloc[240]
oneImage = np.array(oneImage)
oneImage = oneImage[1:]
oneImageArray = spacialExtractor(oneImage, 28)

# Creating the classIndexToAscii array.

asciiValues = np.array([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 100, 101, 102, 103, 104, 110, 113, 114, 116])
asciiValues = np.reshape(asciiValues, (len(asciiValues), 1))
classLabels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46])
classLabels = np.reshape(classLabels, (len(classLabels), 1))
classIndexToAscii = np.concatenate((classLabels, asciiValues), axis = 1)
'''