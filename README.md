# CNN-to-recognise-handwritten-characters

This is a cnn coded from scratch to recognise handwritten characters from the EMNIST dataset. 'Notes' gives a complete description of the project.

The input images are saved as a csv file where each row is a different image, and each column is a different pixel of the flattened image, with the 
first column being the class index.

How to run:
Open 'handWrittenLettersRecognition.py' and run it. This initialises the model as an object with the desired parameters, opens the data, trains the 
model and saves the trained weights and biases to separate csv files. It can also test the accuracy of the model.
