import math
import random
from mnist import MNIST
from sigmoid import sigmoid
import numpy as np

def predict(X):
  X = np.array(X)
  X = X / 256.0

  learning_rate = 0.001

  m = 60000
  input_layer_size = 784
  hidden_layer_size = 25
  output_size = 10
  epsilon_init = 0.12

  num_thetas = (input_layer_size + 1) * hidden_layer_size + (hidden_layer_size + 1) * output_size

  f = open("weights.txt", "r")
  theta = []
  if f.mode == "r":
    theta = f.read().split(',')
    for j, value in enumerate(theta):
      loc = value.find('e')
      if loc != -1:
        exp = int(value[loc + 1 :])
        value = value[0 : loc]
        value = float(value) * math.pow(10, exp)
      theta[j] = float(value)

  lambda_val = 0.1

  theta1 = theta[0:(input_layer_size + 1) * hidden_layer_size]
  theta1 = np.reshape(theta1, (hidden_layer_size, input_layer_size + 1))
  theta2 = theta[(input_layer_size + 1) * hidden_layer_size:]
  theta2 = np.reshape(theta2, (int(len(theta2) / (hidden_layer_size + 1)), hidden_layer_size + 1))

  MAX_EX = 1000

  m = min(len(X), MAX_EX)

  J = 0

  X = np.array(X.copy())

  h = np.zeros((len(X), len(theta2)))

  predictions = np.zeros((len(X), 1))

  for ex in range(len(X)):

    if ex == m:
      break

    a1 = X[ex]
    a1 = np.insert(a1, 0, 1)

    # forward propagation
    a2 = np.zeros((len(theta1), 1))
    z2 = a2.copy()
    for i in range(len(theta1)):
      params = theta1[i]
      z2[i] = a1.dot(params.transpose())
      a2[i] = sigmoid(z2[i])
    
    a2_bias = a2.copy()
    a2_bias = np.insert(a2, 0, 1)

    output = np.zeros((len(theta2), 1))
    z3 = output.copy()

    for i in range(len(theta2)):
      params = theta2[i]
      z3[i] = a2_bias.dot(params.transpose())
      output[i] = sigmoid(z3[i])
    
    h[ex] = output.transpose()

    predictions[ex] = output.argmax() # index == value
  
  return predictions

mndata = MNIST("samples")
X, y = mndata.load_testing()

successes = 0

for i in range(len(X)):
  # print("Test input: {}".format([test]))
  # print(mndata.display(X[i]))
  prediction = predict([X[i]])[0]
  if prediction == y[i]:
    successes += 1

  if i % 100 == 0:
    print("Prediction: {}".format(prediction))
    print("Actual: {}".format(y[i]))
    print("Accuracy: {}".format(successes / (i + 1)))

