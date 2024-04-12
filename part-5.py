from math import log
from scipy.io import loadmat
from itertools import islice

import numpy as np

#This is the implementation of the gradient of the cost function for b (db) and w (dw)
def backprop(y, p, xi):
  db = y-p
  dw = np.matmul(db, xi.T)
  return db, dw

def update_parameters(w, b, db, dw, alpha):
  w = w - alpha * dw
  b = b - alpha * db
  return w, b

#Function to split data into batchs efficiently
def batch_maker(data: dict, SIZE=50):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

#
for batch in batch_maker(M):
   