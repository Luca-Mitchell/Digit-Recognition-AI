from flask import Flask, render_template, redirect, url_for, jsonify, request
import neuralnetwork as nn

i = 0
trainingLabels, trainingImages, testingLabels, testingImages = nn.getData('mnist_train.csv', 'mnist_test.csv')
iterations = 0
alpha = 0
output = []
predictions = []

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prev', methods=['POST'])
def prev():
    global i
    i -= 1
    return redirect(url_for('index'))

@app.route('/next', methods=['POST'])
def next():
    global i
    i += 1
    return redirect(url_for('index'))

@app.route('/getPixelsAsJSON', methods=['GET'])
def getPixelsAsJSON():
    pixels = testingImages.T[i].tolist()
    return jsonify({'pixels': pixels})

@app.route('/setLearningParams', methods=['POST'])
def setLearningParams():
    global iterations, alpha
    data = request.json
    iterations = int(float(data['iterations']))
    alpha = float(data['alpha'])
    return redirect(url_for('index'))

def cleanList(x):
    return [round(float(y), 2) for y in x]

@app.route('/genParams', methods=['GET'])
def genParams():
    global output, predictions
    L1W, L1B, L2W, L2B, testAccuracy, trainAccuracy = nn.gradientDescent(trainingLabels, trainingImages, testingLabels, testingImages, iterations, alpha)
    _, _, _, output = nn.forwardProp(L1W, L1B, L2W, L2B, testingImages)
    predictions = nn.getPredictions(output).tolist()
    output = output.T.tolist()
    return jsonify({'test': cleanList(testAccuracy), 'train': cleanList(trainAccuracy), 'iterations': list(range(1, iterations+1))})

@app.route('/getPredictionData', methods=['GET'])
def getPredictionData():
    predicted = ''
    confidence = ''
    actual = testingLabels[i].tolist()

    if len(predictions) != 0 and len(output) != 0:
        predicted = predictions[i]
        confidence = round(max(output[i])*100, 2)

    return jsonify({'predicted': predicted, 'confidence': confidence, 'actual': actual})

if __name__ == '__main__':
    app.run(debug=True)
