import math
import random
import numpy as np
from sigmoid import sigmoid, sigmoid_gradient
from mnist import MNIST


def cost(theta, input_layer_size, hidden_layer_size, X, y, lambda_value):
  theta1 = theta[0:(input_layer_size + 1) * hidden_layer_size]
  theta1 = np.reshape(theta1, (hidden_layer_size, input_layer_size + 1))
  theta2 = theta[(input_layer_size + 1) * hidden_layer_size:]
  theta2 = np.reshape(theta2, (int(len(theta2) / (hidden_layer_size + 1)), hidden_layer_size + 1))

  MAX_EX = 60000

  m = min(len(X), MAX_EX)

  J = 0

  big_delta1 = np.zeros(theta1.shape)
  big_delta2 = np.zeros(theta2.shape)

  theta1_grad = np.zeros(theta1.shape)
  theta2_grad = np.zeros(theta2.shape)

  output_val = 0

  X = np.array(X.copy())

  h = np.zeros((len(X), len(theta2)))

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
    
    # compute delta2, delta3
    y_new = np.zeros((len(theta2), 1))
    y_new[y[ex]] = 1

    delta3 = output - y_new

    theta2_nobias = theta2.copy()
    theta2_nobias = np.delete(theta2_nobias, 0, axis=1)
    delta2 = (theta2_nobias.transpose().dot(delta3) * a2) * sigmoid_gradient(z2)
    
    # compute big delta2, big delta1
    big_delta2 = big_delta2 + a2_bias.transpose() * delta3
    big_delta1 = big_delta1 + a1.transpose() * delta2

  # delta regularization
  D2 = np.zeros(theta2.shape)
  D2 = (big_delta2 + lambda_value * theta2)
  
  for r, row in enumerate(D2):
    D2[r][0] = big_delta2[r][0]

  D2 = D2 / m

  D1 = np.zeros(theta1.shape)
  D1 = (big_delta1 + lambda_value * theta1)

  for r, row in enumerate(D1):
    D1[r][0] = big_delta1[r][0]

  D1 = D1 / m

  # set new grads
  theta2_grad = D2
  theta1_grad = D1

  # calculate unregularized cost

  for i in range(m):
    actual = np.zeros((len(theta2), 1))
    actual[y[i]] = 1
    for k in range(len(theta2)):
      J += (-actual[k] * math.log(h[i][k]) - (1 - actual[k]) * math.log(1 - h[i][k]))[0]
    # print("Actual: {}".format(actual.transpose()))
    # print("Hypothesis: {}".format(h[i]))
    if i == 98:
      print("SAME")
      print("Actual: {}".format(y[i]))
      print("Prediction: {}".format(h[i].argmax()))
      print("Output value: {}".format(h[i].max()))
      output_val = h[i].max()

    if i == 99:
      print("DIFFERENT")
      print("Actual: {}".format(y[i]))
      print("Prediction: {}".format(h[i].argmax()))
      print("Output value: {}".format(h[i].max()))

  J = J / m

  # regularize cost
  cost_reg = 0

  for r in range(len(theta1)):
    for c in range(len(theta1[r])):
      if c != 0:
        cost_reg += theta1[r][c] * theta1[r][c]

  for r in range(len(theta2)):
    for c in range(len(theta2[r])):
      if c != 0:
        cost_reg += theta2[r][c] * theta2[r][c]
  
  cost_reg = lambda_value / (2 * m) * cost_reg

  J += cost_reg

  grad = np.concatenate((theta1_grad.flatten(),theta2_grad.flatten()))

  return (output_val, J, grad)