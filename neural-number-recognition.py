import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import numpy as np
from scipy.special import softmax
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Input
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import set_random_seed
#from tensorflow.config.experimental import enable_op_determinism

#enable_op_determinism()
set_random_seed(1)

data = loadmat('mnist_all.mat')
data.keys()

# 1-1

fig, ax = plt.subplots(2, 5)
for i in range(10):
  ax[i % 2][i % 5].imshow(data[f"train{i}"][0].reshape((28,28)), cmap='gray')
  ax[i % 2][i % 5].set_title(f"Number {i}")
plt.show()

#Cleaning the data
#Taking too long to run
"""
def clean(D):
  df = []
  for i in range(0,10):
    dict = {"pixel" + str(j) : [D[f"train{i}"][k][j] for k in range(len(D[f"train{i}"]))] for j in range(28*28)}
    dict['Y'] = [i] * len(dict['pixel0'])
    df.append(pd.DataFrame(dict))
  return pd.concat(df)

data_clean = clean(data)
data_clean
"""

def clean(D):
  d = {}
  for i in range(0, 10):
    d["train" + str(i)] = [elem / 255 for elem in D["train" + str(i)] for i in range(0,10)]
  return d
data_clean = clean(data)

# 1-2

from math import log

def softmax(x): #numerically stable softmax
  x = x - np.max(x)
  return np.exp(x) / np.sum(np.exp(x))

def ReLU(x):
  return [max(0, elem) for elem in x]

def forward(X, w, b):
  return np.matmul(X, w) + b

def predict(X, w, b):
  return softmax(forward(X, w, b))

def cross_entropy(y, p):
  return sum([-y[i] * log(p[i]) - (1-y[i]) * log(1-p[i])])

w1 = np.random.rand(28*28, 9)
b1 = np.random.rand(9)

print(predict(np.reshape(data_clean['train0'][0], (28*28)), w1, b1), "\nThis is the predicted outputs for an untrained model")

# 1-3

#This is the implementation of the gradient of the cost function for b (db) and w (dw)
def backprop(y, p, xi):
  db = y-p
  dw = np.matmul(np.array([xi]).T, np.array([db]))
  return db, dw

def update_parameters(w, b, db, dw, alpha):
  w = w - alpha * dw
  b = b - alpha * db
  return w, b

# 1-4

def create_dataframe(data, name):
    df = []
    for i in range(10):
        cur_df = pd.DataFrame({"pixel" + str(j) : [elem[j] / 255 for elem in data[name+str(i)]] for j in range(784)})
        cur_df['Y'] = [i for j in range(len(cur_df))]
        df.append(cur_df)
    return pd.concat(df).reset_index(drop=True)                                     

train = create_dataframe(data, 'train')
test = create_dataframe(data, 'test')

def one_hot(x):
  return np.array([int(i == x) for i in range(10)])

def numerical_approximation_W(x, w, b, y, h):
  a, b = np.shape(w)
  approximations = [[0 for i in range(b)] for i in range(a)]
  for i in range(a):
    for j in range(b):
      h_matrix = np.zeros(np.shape(w))
      h_matrix[i][j] += h
      approximations[i][j] = (cross_entropy(y, predict(x, w + h_matrix, b)) - cross_entropy(y, predict(x, w - h_matrix, b))) / (2 * h)
  return approximations

def numerical_approximation_b(x, w, b, y, h):
  return (cross_entropy(y, predict(x, w, b + h)) - cross_entropy(y, predict(x, w, b-h))) / (2 * h)

def run(X, Y, h=0.01, iterations = 100):
  w = np.random.rand(28*28, 10)
  b = np.random.rand(10)
  current_dw = 0
  current_db = 0
  batch_counter = 0
  estimation_vs_numerical = {'w' : [], 'b' : []}
  
  for i in range(iterations):
    x_i = X.loc[i].to_numpy()
    y_pred = predict(x_i, w, b)
    y = one_hot(Y[i])
    db, dw = backprop(y, y_pred, x_i)
    estimation_vs_numerical['w'].append((dw - numerical_approximation_W(x_i, w, b, y, h)).flatten())
    estimation_vs_numerical['b'].append((db - numerical_approximation_b(x_i, w, b, y, h)).flatten())
    w, b = update_parameters(w, b, db, dw, 0.000001)
  return estimation_vs_numerical
    
    
X = train.sample(frac=1).reset_index(drop=True)
Y = X['Y']
X = X.drop(columns=['Y'])
estimation_vs_numerical = run(X, Y)

def combine(L):
  new_L = []
  for elem in [list(elem) for elem in L]:
    new_L += elem
  return new_L

fig, ax = plt.subplots(2)
ax[0].hist(combine(estimation_vs_numerical['w']))
ax[1].hist(combine(estimation_vs_numerical['b']))
plt.show()

## 1-5
# following two functions are from https://jaykmody.com/blog/stable-softmax/
def log_softmax(x):
    # assumes x is a vector
    x_max = np.max(x)
    return x - x_max - np.log(np.sum(np.exp(x - x_max)))

def cross_entropy_num_safe(y_hat, y_true):
    return -log_softmax(y_hat)[y_true]

# from itertools import islice
# #Function to split data into batchs efficiently
# def batch_maker(data: dict, SIZE=50):
#     it = iter(data)
#     for i in range(0, len(data), SIZE):
#         yield {k:data[k] for k in islice(it, SIZE)}

#
def mini_batch_gradient_descent(X, Y, alpha=0.01, SIZE=50):
  w = np.random.rand(28*28, 10)
  b = np.random.rand(10)
  n = len(X)//SIZE
  X_split = np.array_split(X, n)
  Y_split = np.array_split(Y, n)

  for i in range(n):
    X_i = X_split[i].reset_index(drop=True)
    Y_i = Y_split[i].reset_index(drop=True)
    for j in range(SIZE):
      x_i = X_i.loc[j].to_numpy()
      y_pred = predict(x_i, w, b)
      y = one_hot(Y_i[j])
      db, dw = backprop(y, y_pred, x_i)
      w, b = update_parameters(w, b, db, dw, alpha)
  
  return w, b

def validate_mbgd(X, Y, w, b):
  X = X.reset_index(drop=True)
  Y = Y.reset_index(drop=True)

  out = 0
  for i in range(len(X)):
    x_i = X.loc[i].to_numpy()
    y_hat = forward(x_i, w, b)
    y = Y[i]
    out += cross_entropy_num_safe(y_hat, y)
    
  return out/len(X)

x_train = train.sample(frac=1).reset_index(drop=True)
y_train = x_train['Y']
x_train = x_train.drop(columns=['Y'])

split_value = int(len(x_train)*0.1)

x_val_5 = x_train.iloc[:split_value,:].reset_index(drop=True)
y_val_5 = y_train.iloc[:split_value,].reset_index(drop=True)

x_train_5 = x_train.iloc[split_value:,:].reset_index(drop=True)
y_train_5 = y_train.iloc[split_value:,].reset_index(drop=True)

w, b = mini_batch_gradient_descent(x_train_5, y_train_5)
accuracy = validate_mbgd(x_val_5, y_val_5, w, b)
print("1-5", w, b, accuracy)

exit() #TO REMOVE
# 1-6

# 1-7 and 1-8

# load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#split the training set into a new training set and a validation set
n_val = int(len(x_train)*0.1)

x_val = x_train[:n_val]
y_val = y_train[:n_val]

x_train_nv = x_train[n_val:]
y_train_nv = y_train[n_val:]

def my_model_mbgd(shape, n, classes, learning_rate, x_train, y_train, epochs, batch_size, x_val, y_val):
    """
    Initializes, compiles, and fits a model.
    
    Parameters:
    - shape: tuple, the shape of the input images ((28,28) for minst)
    - n: int, the number of nodes in the hidden layer
    - classes: int, the number of classes (10 for minst)
    - learning_rate: float, the learning rate
    - x_train: numpy.ndarray
    - y_train: numpy.ndarray
    - epochs: int
    - batch_size: int
    - x_val: numpy.ndarray
    - y_val: numpy.ndarray
      
    Returns:
    - The fitted model and the history object.
    """
    
    # initialize model
    model = Sequential([
        Input(shape=shape),
        Flatten(),
        Dense(n, activation="tanh", name="hidden"), # new layer
        Dense(classes)
    ])

    # compile model
    model.compile(optimizer=SGD(learning_rate=learning_rate),
                  loss=SparseCategoricalCrossentropy(from_logits=True), # from_logits=True applies softmax to loss
                  metrics=["accuracy"]
                 )

    # fit model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    
    return model, history

model, history = my_model_mbgd(shape=(28,28),n=300,classes=10,learning_rate=0.01,x_train=x_train_nv,y_train =y_train_nv,epochs=50,batch_size=50,x_val=x_val,y_val=y_val)

test_loss, test_acc = model.evaluate(x_test, y_test)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(14, 5))

# Plot training and validation accuracy per epoch
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot training and validation loss per epoch
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

predictions = model.predict(x_test)
pred_class = np.argmax(predictions, axis=1)

pred_correct = np.where(pred_class == y_test)[0]
pred_incorrect = np.where(pred_class != y_test)[0]

plt.figure(figsize=(10, 5))
for i, correct in enumerate(pred_correct[:20]):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_test[correct].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {pred_class[correct]}, True: {y_test[correct]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
for i, incorrect in enumerate(pred_incorrect[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[incorrect].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {pred_class[incorrect]}, True: {y_test[incorrect]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 1-9

weights = model.layers[1].weights[0]

interesting_w = [91, 220]

interesting_values = [np.reshape(weights[:, interesting_w[0]], 28*28), np.reshape(weights[:, interesting_w[1]], 28*28)]
interesting_values = np.reshape(interesting_values, 2*28*28)
interesting_limit = max([abs(min(interesting_values)), abs(max(interesting_values))]) # for obtaining a colour bar centred at zero

plt.figure(figsize = (14, 5))

plt.subplot(1, 2, 1)
plt.imshow(np.reshape(weights[:, interesting_w[0]], (28, 28)), cmap='coolwarm', vmin = -interesting_limit, vmax = interesting_limit)
plt.title(interesting_w[0])
plt.axis('off')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(np.reshape(weights[:, interesting_w[1]], (28, 28)), cmap='coolwarm', vmin = -interesting_limit, vmax = interesting_limit)
plt.axis('off')
plt.title(interesting_w[1])

plt.show()

interesting_weights = model.layers[2].weights[0]
interesting_weights = [interesting_weights[interesting_w[0], :].numpy(), interesting_weights[interesting_w[1], :].numpy()]

for k in range(len(interesting_weights)):
    print("Output weights from hidden neuron " + str(interesting_w[k]) + ":")
    for i in range(10):
        print(str(i) + ": " + str(interesting_weights[k][i]))
    print("")
