from NeuralNetworkHelperFunctions import getData, getPredictions, getAccuracy
from NeuralNetwork import gradientDescent, forwardPropogate

trainingLabels, trainingImages, testingLabels, testingImages = getData("data/mnist_train.csv", "data/mnist_test.csv")

while True:

    iterations = int(input("iterations >>> "))
    alpha = float(input("alpha >>> "))

    L1w, L2w, L3w, L4w, L1b, L2b, L3b, L4b = gradientDescent(trainingLabels, trainingImages, iterations, alpha)
    _, _, _, _, _, _, _, output = forwardPropogate(L1w, L2w, L3w, L4w, L1b, L2b, L3b, L4b, testingImages)
    predictions = getPredictions(output)
    accuracy = getAccuracy(predictions, testingLabels)
    print(accuracy)