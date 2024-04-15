## Imports for early parts
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

## 1-1

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(y_train[range(10)])

fig, ax = plt.subplots(nrows = 10, ncols = 10, dpi = 200)
for digit in range(10):
    for counter in range(10): # display each digit 10 times
        index = np.where(y_train == digit)[0][counter]
        ax[counter % 10][digit % 10].imshow(x_train[index].reshape((28,28)), cmap='gray')
        ax[counter % 10][digit % 10].axis("off")
plt.savefig(fname = "figures/mnist-samples.png", format = "png")
plt.show()

## 1-2

from math import log

def softmax(x):
  e_x = np.exp(x)
  return e_x / e_x.sum()

def ReLU(x):
  return [max(0, elem) for elem in x]

def forward(X, w, b):
  return np.matmul(X, w) + b

def predict(X, w, b):
  return softmax(forward(X, w, b))

def cross_entropy(y, p):
  return sum([-y[i] * log(p[i]) - (1-y[i]) * log(1-p[i])])

w1 = np.random.rand(28*28, 10)
b1 = np.random.rand(10)

print("The untrained model makes the following prediction for any input image:")
for i in range(len(b1)):
    print(str(i) + ": " + str(predict(np.reshape(x_train[0], (28*28)), w1, b1)[i]))

## 1-3

# This is the implementation of the gradient of the cost function for b (db) and w (dw)
def backprop(y, p, xi):
  db = y-p
  dw = np.matmul(np.array([xi]).T, np.array([db]))
  return db, dw

def update_parameters(w, b, db, dw, alpha):
  w = w - alpha * dw
  b = b - alpha * db
  return w, b

## 1-4

def one_hot(x): # one-hot vector encoding
  return np.array([int(i == x) for i in range(10)])

def numerical_approximation_W(x, w, b, y, h):
  a, b = np.shape(w) # number of rows and columns of the matrix w
  approximations = [[0 for i in range(b)] for i in range(a)] # initialize to zero matrix
  for i in range(a):
    for j in range(b):
      h_matrix = np.zeros(np.shape(w))
      h_matrix[i][j] += h # perturbation in the (i, j) direction
      approximations[i][j] = (cross_entropy(y, predict(x, w + h_matrix, b)) - cross_entropy(y, predict(x, w - h_matrix, b))) / (2 * h)
  return approximations # two-sided approximation

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
    x_i = np.reshape(X[i], 28*28)
    y_pred = predict(x_i, w, b)
    y = one_hot(Y[i])
    db, dw = backprop(y, y_pred, x_i)
    estimation_vs_numerical['w'].append((dw - numerical_approximation_W(x_i, w, b, y, h)).flatten())
    estimation_vs_numerical['b'].append((db - numerical_approximation_b(x_i, w, b, y, h)).flatten())
    w, b = update_parameters(w, b, db, dw, 0.000001)
  return estimation_vs_numerical

estimation_vs_numerical = run(x_train, y_train)

def combine(L):
  new_L = []
  for elem in [list(elem) for elem in L]:
    new_L += elem
  return new_L

fig, ax = plt.subplots(2)
ax[0].hist(combine(estimation_vs_numerical['w']))
ax[1].hist(combine(estimation_vs_numerical['b']))
plt.savefig(fname = "figures/compare-gradients.png", format = "png")
plt.show()

## 1-5

## 1-6

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
  n = len(X)//SIZE # floor division
  X_split = np.array_split(X, n)
  Y_split = np.array_split(Y, n)

  for i in range(n):
    X_i = X_split[i]
    Y_i = Y_split[i]
    dbTotal = 0
    dwTotal = 0
    for j in range(SIZE):
      x_i = np.reshape(X_i[j], 28*28)
      y_pred = predict(x_i, w, b)
      y = one_hot(Y_i[j])
      db, dw = backprop(y, y_pred, x_i)
      dbTotal += db
      dwTotal += dw
    w, b = update_parameters(w, b, dbTotal/SIZE, dwTotal/SIZE, alpha)
  
  return w, b

def validate_mbgd(X, Y, w, b):

  out = 0
  for i in range(len(X)):
    x_i = np.reshape(X[i], 28*28)
    y_hat = forward(x_i, w, b)
    y = Y[i]
    out += cross_entropy_num_safe(y_hat, y)
    
  return out/len(X)

split_value = int(len(x_train)*0.1)

x_val_5 = x_train[:split_value]
y_val_5 = y_train[:split_value]

x_train_5 = x_train[split_value:]
y_train_5 = y_train[split_value:]

w, b = mini_batch_gradient_descent(x_train_5, y_train_5)
avg_loss = validate_mbgd(x_val_5, y_val_5, w, b)
print("1-5", w, b, avg_loss)

## Imports for later parts

from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import set_random_seed
from tensorflow.config.experimental import enable_op_determinism

enable_op_determinism()
set_random_seed(1)

## 1-7

def my_model(shape, n, classes, x_train, y_train):
    """
    Initializes, compiles, and fits a model.
    
    Parameters:
    - shape: tuple, the shape of the input images ((28,28) for minst)
    - n: int, the number of nodes in the hidden layer
    - classes: int, the number of classes (10 for minst)
    - x_train: numpy.ndarray
    - y_train: numpy.ndarray
      
    Returns:
    - The fitted model.
    """
    
    # initialize model
    model = Sequential([
        Input(shape=shape),
        Flatten(),
        Dense(n, activation="tanh"), # new layer
        Dense(classes)
    ])

    # compile model
    model.compile(optimizer="adam",
                  loss=SparseCategoricalCrossentropy(from_logits=True), # from_logits=True applies softmax to loss
                  metrics=["accuracy"]
                 )

    # fit model
    model.fit(x_train, y_train)
    
    return model

model_0 = my_model((28,28),300,10,x_train,y_train)

test_loss_0, test_acc_0 = model_0.evaluate(x_test, y_test)

## 1-8

#split the training set into a new training set and a validation set
n_val = int(len(x_train)*0.1)

x_val = x_train[:n_val]
y_val = y_train[:n_val]

x_train_nv = x_train[n_val:]
y_train_nv = y_train[n_val:]

def my_model_mbgd(
    x_train,
    y_train,
    x_val,
    y_val,
    learning_rate,
    batch_size,
    epochs = 50,
    shape = (28, 28),
    hl_size = 300,
    n_classes = 10
):
    """
    Initializes, compiles, and fits a model.
    
    Parameters:
    - x_train: numpy.ndarray
    - y_train: numpy.ndarray
    - x_val: numpy.ndarray
    - y_val: numpy.ndarray
    - learning_rate: float, the learning rate
    - batch_size: int
    - epochs: int, the number of training epochs (default: 50)
    - shape: tuple, the shape of the input images (default: (28, 28))
    - hl_size: int, the number of nodes in the hidden layer (default: 300)
    - n_classes: int, the number of classes (default: 10)
      
    Returns:
    - The fitted model and the history object.
    """
    
    # initialize model

    model = Sequential(
        [
            Input(shape = shape),
            Flatten(),
            Dense(hl_size, activation="tanh"), # new layer
            Dense(n_classes)
        ]
    )

    # compile model
    model.compile(
        optimizer=SGD(learning_rate=learning_rate),
        loss=SparseCategoricalCrossentropy(from_logits=True), # from_logits=True applies softmax to loss
        metrics=["accuracy"]
    )

    # fit model
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val)
    )
    
    return model, history

model, history = my_model_mbgd(
    x_train=x_train_nv,
    y_train=y_train_nv,
    x_val=x_val,
    y_val=y_val,
    learning_rate=0.01,
    batch_size=50,
    epochs = 30
)
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
plt.savefig(fname = "figures/training-validation.png", format = "png")
plt.show()

predictions = model.predict(x_test)
pred_class = np.argmax(predictions, axis=1)

pred_correct = np.where(pred_class == y_test)[0]
pred_incorrect = np.where(pred_class != y_test)[0]

plt.figure(figsize=(10, 5))
for i, correct in enumerate(pred_correct[:20]):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_test[correct].reshape(28, 28), cmap='gray')
    plt.title(f"Correct: {y_test[correct]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(fname = "figures/correctly-classified.png", format = "png")
plt.show()

plt.figure(figsize=(10, 5))
for i, incorrect in enumerate(pred_incorrect[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[incorrect].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {pred_class[incorrect]}, True: {y_test[incorrect]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(fname = "figures/incorrectly-classified.png", format = "png")
plt.show()

## 1-9

weights = model.layers[1].weights[0]

interesting_indices = [295, 224]
interesting_values = [
    np.reshape(weights[:, interesting_indices[0]], 28*28),
    np.reshape(weights[:, interesting_indices[1]], 28*28)
]
interesting_values = np.reshape(interesting_values, 2*28*28)
interesting_limit = max(
    [abs(min(interesting_values)), abs(max(interesting_values))]
) # for obtaining a colour bar centred at zero

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.imshow(
    np.reshape(weights[:, interesting_indices[0]], (28, 28)),
    cmap='coolwarm',
    vmin = -interesting_limit,
    vmax = interesting_limit
)
plt.title(interesting_indices[0])
plt.axis('off')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(
    np.reshape(weights[:, interesting_indices[1]], (28, 28)),
    cmap='coolwarm',
    vmin = -interesting_limit,
    vmax = interesting_limit
)
plt.axis('off')
plt.title(interesting_indices[1])
plt.savefig(fname = "figures/interesting-weights.png", format = "png")
plt.show()

interesting_weights = model.layers[2].weights[0]
interesting_weights = [
    interesting_weights[interesting_indices[0], :].numpy(),
    interesting_weights[interesting_indices[1], :].numpy()
]
for k in range(2):
    print("Output weights from hidden neuron " + str(interesting_indices[k]) + ":")
    for i in range(10):
        print(str(i) + ": " + str(interesting_weights[k][i]))
    print("")
