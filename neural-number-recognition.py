## Imports for early parts
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

np.random.seed(2009) # makes 1-1 to 1-6 reproducible

## version info
import sys
print(sys.version)
import tensorflow.version
print(tensorflow.version.VERSION)
print(np.version.version)

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
  return sum([-y[i] * log(p[i]) - (1-y[i]) * log(1-p[i]) for i in range(len(y))])

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

# following two functions are from https://jaykmody.com/blog/stable-softmax/
def log_softmax(x):
    # assumes x is a vector
    x_max = np.max(x)
    return x - x_max - np.log(np.sum(np.exp(x - x_max)))

def cross_entropy_num_safe(y_hat, y_true):
    return -log_softmax(y_hat)[y_true]

def log_loss(y_hat, y_true, THRESHOLD=1):
   entropy = cross_entropy_num_safe(y_hat, y_true)
   if entropy >= THRESHOLD:
      return 0 #misclassified
   else:
      return 1 #properly classified

def validate_mbgd(X, Y, w, b):

  out = 0
  for i in range(len(X)):
    x_i = np.reshape(X[i], 28*28)
    y_hat = forward(x_i, w, b)
    y = Y[i]
    out += log_loss(y_hat, y)
    
  return out/len(X)

#
def mini_batch_gradient_descent(X, Y, alpha=0.01, SIZE=50):
  split_value = int(len(X)*0.1)

  x_val = X[:split_value]
  y_val = Y[:split_value]

  x_train = X[split_value:]
  y_train = Y[split_value:]

  w, b = np.random.rand(28*28, 10), np.random.rand(10)
  n = len(X)//SIZE # floor division
  X_split = np.array_split(x_train, n)
  Y_split = np.array_split(y_train, n)

  learning_curve = {'data': [], 'labels': []}
  learning_curve['data'].append(validate_mbgd(x_val, y_val, w, b))

  for i in range(n):
    X_i = X_split[i]
    Y_i = Y_split[i]
    dbTotal = 0
    dwTotal = 0
    for j in range(len(X_i)):
      x_j = np.reshape(X_i[j], 28*28)
      y_pred = predict(x_j, w, b)
      loss = log_loss(y_pred, Y_i[j])
      if loss == 0: #if predictor isn't right, we find the gradient 
        y = one_hot(Y_i[j])
        db, dw = backprop(y, y_pred, x_j)
        dbTotal += db
        dwTotal += dw
    w, b = update_parameters(w, b, dbTotal/SIZE, dwTotal/SIZE, alpha)
    learning_curve['data'].append(validate_mbgd(x_val, y_val, w, b))
    learning_curve['labels'].append((w,b))
  
  return learning_curve

def mini_batch_gradient_descent_2(x_train, y_train, w, b, alpha=0.01, SIZE=50):
  n = len(x_train)//SIZE # floor division
  X_split = np.array_split(x_train, n)
  Y_split = np.array_split(y_train, n)

  for i in range(n):
    X_i = X_split[i]
    Y_i = Y_split[i]
    dbTotal = 0
    dwTotal = 0
    for j in range(len(X_i)):
      x_j = np.reshape(X_i[j], 28*28)
      y_pred = predict(x_j, w, b)
      loss = log_loss(y_pred, Y_i[j])
      if loss == 0: #if predictor isn't right, we find the gradient 
        y = one_hot(Y_i[j])
        db, dw = backprop(y, y_pred, x_j)
        dbTotal += db
        dwTotal += dw
    w, b = update_parameters(w, b, dbTotal/SIZE, dwTotal/SIZE, alpha)
  
  return w, b

split_value = int(len(x_train)*0.1)

x_val = x_train[:split_value]
y_val = y_train[:split_value]

x_train_5 = x_train[split_value:]
y_train_5 = y_train[split_value:]

w, b = np.random.rand(28*28, 10), np.random.rand(10)
learning_curve = {'data': [], 'labels': []}
learning_curve['data'].append(validate_mbgd(x_val, y_val, w, b))
learning_curve['labels'].append((w,b))
EPOCHES = 10
for i in range(EPOCHES):
  w, b = mini_batch_gradient_descent_2(x_train_5, y_train_5, w, b)
  learning_curve['data'].append(validate_mbgd(x_val, y_val, w, b))
  learning_curve['labels'].append((w,b))

data = learning_curve['data']
fig = plt.figure(clear=True)
ax = fig.add_subplot(111)
ax.plot(EPOCHES+1, data)
labels = learning_curve['labels']
# for i in range(0, len(labels), 100): #attempt to plot the weights and biases on the graph
#    wb = labels[i]
#    ax.annotate('%sX+%s' % wb, xy=(i,data[i]), textcoords='data')

plt.grid()
plt.savefig(fname = "figures/1-5-learning-curve.png", format = "png")
plt.show()

## 1-6

## Imports for later parts

from tensorflow.keras.layers import Dense, Flatten, Input, MultiHeadAttention, Reshape, LayerNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
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
    inputs = Input(shape=shape)
    x = Flatten()(inputs)
    x = Dense(n, activation="tanh")(x)
    outputs = Dense(classes)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
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
    
    # Initialize model
    inputs = Input(shape=shape)
    x = Flatten()(inputs)
    x = Dense(hl_size, activation="tanh")(x)
    outputs = Dense(n_classes)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(optimizer=SGD(learning_rate=learning_rate),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    
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

weights = model.layers[2].weights[0]

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

interesting_weights = [
    weights[interesting_indices[0], :].numpy(),
    weights[interesting_indices[1], :].numpy()
]
for k in range(2):
    print("Output weights from hidden neuron " + str(interesting_indices[k]) + ":")
    for i in range(10):
        print(str(i) + ": " + str(interesting_weights[k][i]))
    print("")

## 2 [implementation]

def my_model_attn(shape, n, classes, learning_rate, x_train, y_train, epochs, batch_size, x_val, y_val):
    """
    Initializes, compiles, and fits a model with an attention mechanism layer.
    
    Parameters:
    - shape: tuple, the shape of the input images ((28,28) for mnist)
    - n: int, the number of nodes in the hidden layer
    - classes: int, the number of classes (10 for mnist)
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
    
    # model layers
    inputs = Input(shape=shape)
    x = Flatten()(inputs)
    x = LayerNormalization()(x) # layer normalization
    x = Reshape(shape)(x) # reshaping in preparation for multi-headed attention
    x = MultiHeadAttention(num_heads=2, key_dim=14)(x, x) # multi-headed attention layer
    x = Flatten()(x)
    x = Dense(n, activation="tanh")(x)
    outputs = Dense(classes)(x)
    
    model = Model(inputs=inputs, outputs=outputs)

    # compile model
    model.compile(optimizer=SGD(learning_rate=learning_rate),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
                  
    # history object (for plotting accuracy and loss)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    
    return model, history

model_attn, history_attn = my_model_attn(shape=(28,28),n=300,classes=10,learning_rate=0.01,x_train=x_train_nv,y_train =y_train_nv,epochs=39,batch_size=50,x_val=x_val,y_val=y_val)

acc_attn = history_attn.history['accuracy']
val_acc_attn = history_attn.history['val_accuracy']
loss_attn = history_attn.history['loss']
val_loss_attn = history_attn.history['val_loss']
epochs_range_attn = range(1, len(acc_attn) + 1)

acc_attn = history_attn.history['accuracy']
val_acc_attn = history_attn.history['val_accuracy']
loss_attn = history_attn.history['loss']
val_loss_attn = history_attn.history['val_loss']
epochs_range_attn = range(1, len(acc_attn) + 1)

plt.figure(figsize=(14, 5))

# plot training and validation accuracy per epoch
plt.subplot(1, 2, 1)
plt.plot(epochs_range_attn, acc_attn, label='Training Accuracy')
plt.plot(epochs_range_attn, val_acc_attn, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# plot training and validation loss per epoch
plt.subplot(1, 2, 2)
plt.plot(epochs_range_attn, loss_attn, label='Training Loss')
plt.plot(epochs_range_attn, val_loss_attn, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()
