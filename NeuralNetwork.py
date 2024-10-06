import numpy as np
from NeuralNetworkHelperFunctions import oneHotEncode




##### SETUP #####


def initialiseParameters():

    L1w = np.random.randn(256, 784) * np.sqrt(2 / 784)
    L2w = np.random.randn(128, 256) * np.sqrt(2 / 256)
    L3w = np.random.randn(64, 128) * np.sqrt(2 / 128)
    L4w = np.random.randn(10, 64) * np.sqrt(2 / 64)

    L1b = np.random.rand(256, 1) - 0.5
    L2b = np.random.rand(128, 1) - 0.5
    L3b = np.random.rand(64, 1) - 0.5
    L4b = np.random.rand(10, 1) - 0.5

    return L1w, L2w, L3w, L4w, L1b, L2b, L3b, L4b




##### FORWARD PROPOGATION #####

def leakyReLU(z):
    return np.maximum(0.01 * z, z)


def ReLU(z):
    return np.maximum(0, z)


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)


def forwardPropogate(L1w, L2w, L3w, L4w, L1b, L2b, L3b, L4b, inputs):

    L1z = np.dot(L1w, inputs) + L1b
    L1a = leakyReLU(L1z)

    L2z = np.dot(L2w, L1a) + L2b
    L2a = leakyReLU(L2z)

    L3z = np.dot(L3w, L2a) + L3b
    L3a = leakyReLU(L3z)

    L4z = np.dot(L4w, L3a) + L4b
    L4a = softmax(L4z)

    return L1z, L1a, L2z, L2a, L3z, L3a, L4z, L4a




##### BACK PROPOGATION #####


def backPropogate(L1w, L2w, L3w, L4w, L1b, L2b, L3b, L4b, L1z, L1a, L2z, L2a, L3z, L3a, L4z, L4a, inputs, oneHotLabels):
    numOfImages = inputs.shape[1]

    # Layer 4 (Output Layer)
    dL4z = L4a - oneHotLabels  # Error at Layer 4
    dL4w = np.dot(dL4z, L3a.T) / numOfImages  # Gradient for Layer 4 weights
    dL4b = np.sum(dL4z, axis=1, keepdims=True) / numOfImages  # Gradient for Layer 4 biases

    # Layer 3
    dL3a = np.dot(L4w.T, dL4z)  # Propagate the error to Layer 3
    dL3z = dL3a * (L3z > 0)  # ReLU derivative for Layer 3
    dL3w = np.dot(dL3z, L2a.T) / numOfImages  # Gradient for Layer 3 weights
    dL3b = np.sum(dL3z, axis=1, keepdims=True) / numOfImages  # Gradient for Layer 3 biases

    # Layer 2
    dL2a = np.dot(L3w.T, dL3z)  # Propagate the error to Layer 2
    dL2z = dL2a * (L2z > 0)  # ReLU derivative for Layer 2
    dL2w = np.dot(dL2z, L1a.T) / numOfImages  # Gradient for Layer 2 weights
    dL2b = np.sum(dL2z, axis=1, keepdims=True) / numOfImages  # Gradient for Layer 2 biases

    # Layer 1
    dL1a = np.dot(L2w.T, dL2z)  # Propagate the error to Layer 1
    dL1z = dL1a * (L1z > 0)  # ReLU derivative for Layer 1
    dL1w = np.dot(dL1z, inputs.T) / numOfImages  # Gradient for Layer 1 weights
    dL1b = np.sum(dL1z, axis=1, keepdims=True) / numOfImages  # Gradient for Layer 1 biases

    return dL1w, dL2w, dL3w, dL4w, dL1b, dL2b, dL3b, dL4b



def updateParams(L1w, L2w, L3w, L4w, L1b, L2b, L3b, L4b, dL1w, dL2w, dL3w, dL4w, dL1b, dL2b, dL3b, dL4b, alpha):

    L1w -= alpha * dL1w
    L2w -= alpha * dL2w
    L3w -= alpha * dL3w
    L4w -= alpha * dL4w

    L1b -= alpha * dL1b
    L2b -= alpha * dL2b
    L3b -= alpha * dL3b
    L4b -= alpha * dL4b

    return L1w, L2w, L3w, L4w, L1b, L2b, L3b, L4b




##### GRADIENT DESCENT #####


def gradientDescent(trainingLabels, trainingImages, i, alpha):

    L1w, L2w, L3w, L4w, L1b, L2b, L3b, L4b = initialiseParameters()
    oneHotLabels = oneHotEncode(trainingLabels)

    for iteration in range(i):

        L1z, L1a, L2z, L2a, L3z, L3a, L4z, L4a = forwardPropogate(L1w, L2w, L3w, L4w, L1b, L2b, L3b, L4b, trainingImages)
        dL1w, dL2w, dL3w, dL4w, dL1b, dL2b, dL3b, dL4b = backPropogate(L1w, L2w, L3w, L4w, L1b, L2b, L3b, L4b, L1z, L1a, L2z, L2a, L3z, L3a, L4z, L4a, trainingImages, oneHotLabels)
        L1w, L2w, L3w, L4w, L1b, L2b, L3b, L4b = updateParams(L1w, L2w, L3w, L4w, L1b, L2b, L3b, L4b, dL1w, dL2w, dL3w, dL4w, dL1b, dL2b, dL3b, dL4b, alpha)

        print(f"iteration {iteration}")

    return L1w, L2w, L3w, L4w, L1b, L2b, L3b, L4b