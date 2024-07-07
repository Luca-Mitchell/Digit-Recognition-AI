import numpy as np
import pandas as pd

def getData(traingDataPath, testingDataPath):
    trainingData = np.array(pd.read_csv(traingDataPath)).T
    trainingLabels = trainingData[0]
    trainingImages = trainingData[1:trainingData.size] / 255

    testingData = np.array(pd.read_csv(testingDataPath)).T
    testingLabels = testingData[0]
    testingImages = testingData[1:testingData.size] / 255

    return trainingLabels, trainingImages, testingLabels, testingImages

def initParams():
    L1W = np.random.rand(10, 784) - 0.5
    L1B = np.random.rand(10, 1) - 0.5
    L2W = np.random.rand(10, 10) - 0.5
    L2B = np.random.rand(10, 1) - 0.5
    return L1W, L1B, L2W, L2B

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

def forwardProp(L1W, L1B, L2W, L2B, inputs):
    L1Z = np.dot(L1W, inputs) + L1B
    L1A = ReLU(L1Z)
    L2Z = np.dot(L2W, L1A) + L2B
    L2A = softmax(L2Z)
    return L1Z, L1A, L2Z, L2A

def oneHotEncode(labels):
    labelsList = list(labels)
    oneHotLabels = []
    for label in labelsList:
        oneHotLabel = [0] * 10
        oneHotLabel[label] = 1
        oneHotLabels.append(oneHotLabel)
    return np.array(oneHotLabels).T

def backProp(L1Z, L1A, L2A, L2W, inputs, oneHotLabels): # I don't know how this bit works
    numOfImages = inputs.shape[1]

    dL2Z = L2A - oneHotLabels
    dL2W = np.dot(dL2Z, L1A.T) / numOfImages
    dL2B = np.sum(dL2Z, axis=1, keepdims=True) / numOfImages

    dL1A = np.dot(L2W.T, dL2Z)
    dL1Z = dL1A * (L1Z > 0)
    dL1W = np.dot(dL1Z, inputs.T) / numOfImages
    dL1B = np.sum(dL1Z, axis=1, keepdims=True) / numOfImages

    return dL1W, dL1B, dL2W, dL2B

def updateParams(L1W, L1B, L2W, L2B, dL1W, dL1B, dL2W, dL2B, alpha):
    L1W -= alpha * dL1W
    L1B -= alpha * dL1B
    L2W -= alpha * dL2W
    L2B -= alpha * dL2B
    return L1W, L1B, L2W, L2B

def getPredictions(output):
    return np.argmax(output, axis=0)

def getAccuracy(predictions, labels):
    numCorrect = np.sum(predictions == labels)
    percentageCorrect = (numCorrect / labels.shape[0]) * 100
    return percentageCorrect

def gradientDescent(trainingLabels, trainingImages, testingLabels, testingImages, i, alpha):

    L1W, L1B, L2W, L2B = initParams()
    oneHotLabels = oneHotEncode(trainingLabels)

    testDataAccuracyList = []
    trainingDataAccuracyList = []

    for iter in range(i):

        L1Z, L1A, L2Z, L2A = forwardProp(L1W, L1B, L2W, L2B, trainingImages)
        dL1W, dL1B, dL2W, dL2B = backProp(L1Z, L1A, L2A, L2W, trainingImages, oneHotLabels)
        L1W, L1B, L2W, L2B = updateParams(L1W, L1B, L2W, L2B, dL1W, dL1B, dL2W, dL2B, alpha)

        _, _, _, testDataOutput = forwardProp(L1W, L1B, L2W, L2B, testingImages)
        testDataPredictions = getPredictions(testDataOutput)
        testDatAccuracy = getAccuracy(testDataPredictions, testingLabels)

        _, _, _, trainingDataOutput = forwardProp(L1W, L1B, L2W, L2B, trainingImages)
        trainingDataPredictions = getPredictions(trainingDataOutput)
        trainingDatAccuracy = getAccuracy(trainingDataPredictions, trainingLabels)

        testDataAccuracyList.append(testDatAccuracy)
        trainingDataAccuracyList.append(trainingDatAccuracy)
            
    return L1W, L1B, L2W, L2B, testDataAccuracyList, trainingDataAccuracyList