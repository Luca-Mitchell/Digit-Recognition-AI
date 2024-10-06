import numpy as np
import pandas as pd


def getData(traingDataPath, testingDataPath):

    trainingData = np.array(pd.read_csv(traingDataPath))
    trainingData = trainingData.T
    trainingLabels = trainingData[0]
    trainingImages = trainingData[1:trainingData.size] / 255

    testingData = np.array(pd.read_csv(testingDataPath))
    testingData = testingData.T
    testingLabels = testingData[0]
    testingImages = testingData[1:testingData.size] / 255

    return trainingLabels, trainingImages, testingLabels, testingImages


def oneHotEncode(labels):

    labelsList = list(labels)
    oneHotLabels = []

    for label in labelsList:
        oneHotLabel = [0] * 10
        oneHotLabel[label] = 1
        oneHotLabels.append(oneHotLabel)
        
    return np.array(oneHotLabels).T


def getPredictions(output):
    return np.argmax(output, axis=0)


def getAccuracy(predictions, labels):
    numCorrect = np.sum(predictions == labels)
    percentageCorrect = (numCorrect / labels.shape[0]) * 100
    return percentageCorrect
