import math
import random
from sigmoid import sigmoid
from cost_function import cost
from mnist import MNIST
import numpy as np

def train():
  mndata = MNIST("samples")
  X, y = mndata.load_training()
  # index = random.randrange(0, len(images))
  # print(mndata.display(images[index]))
  # print(labels[index])
  X = np.array(X)
  X = X / 256.0

  learning_rate = 0.1
  lambda_val = 0.1

  m = 60000
  input_layer_size = 784
  hidden_layer_size = 25
  output_size = 10
  epsilon_init = 0.12

  num_thetas = (input_layer_size + 1) * hidden_layer_size + (hidden_layer_size + 1) * output_size

  theta = [(random.random() + 0.5) / 100 for i in range(num_thetas)]

  # print("init theta2[0]: {}".format(theta2[0]))

  output_prev = 0

  iterations = 10000
  for i in range(iterations):
    print("===============")
    print("Iteration {}".format(i))

    print("Calculating cost...")
    output, J, grad = cost(theta, input_layer_size, hidden_layer_size, X, y, lambda_val)
    print("Calculated cost: {}".format(J))
    print("Change in value: {}".format(output - output_prev))
    output_prev = output
    theta -= learning_rate * grad

    if i % 5 == 0:
      print("Writing weights")
      with open("weights.txt", "w") as f:
        f.write("%s" % theta[0])
        for j, param in enumerate(theta):
          if j != 0:
            f.write(",%s" % param)

    print("Done Iteration {}".format(i))
    print("===============")

  return theta

print("Begin training")
theta = train()
print("Done training")

with open("weights.txt", "w") as f:
  f.write("%s" % theta[0])
  for j, param in enumerate(theta):
    if j != 0:
      f.write(",%s" % param)
