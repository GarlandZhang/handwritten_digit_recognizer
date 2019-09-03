import math
import numpy as np

def sigmoid(input):
  if isinstance(input, (int, float, np.int32)):
    return 1.0 / (1.0 + math.exp(-input))
  elif isinstance(input, (list, np.ndarray)):
    result = input.copy()
    for i in range(len(result)):
      result[i] = sigmoid(result[i])
    return result
  else:
    print("not valid input: {}, type is: {}".format(input, input.__class__))
    return None

def sigmoid_gradient(input):
  result = sigmoid(input)
  row, col = result.shape
  for r in range(row):
    for c in range(col):
      result[r][c] = result[r][c] * (1 - result[r][c])
  
  return result

# print(sigmoid_gradient(np.array([[2.0,3.0,4.0],[1.0,5.0,7.0]])))