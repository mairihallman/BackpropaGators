# Part 1.2


```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.random import set_seed
```


```python
set_seed(1)
```


```python
# load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```


```python
def my_model(shape, n, x_train, y_train):
    """
    Initializes, compiles, and fits a model.
    
    Parameters:
    - shape: tuple, the shape of the input images ((28,28) for minst)
    - n: int, the number of classes (10 for minst)
    - x_train: numpy.ndarray
    - y_train: numpy.ndarray
      
    Returns:
    - The fitted model.
    """
    
    # initialize model
    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),
        Dense(10, activation="linear")
    ])

    # compile model
    model.compile(optimizer="adam",
                  loss=SparseCategoricalCrossentropy(from_logits=True), # from_logits=True applies softmax to loss
                  metrics=["accuracy"]
                 )

    # fit model
    model.fit(x_train, y_train)
    
    return model
```


```python
model = my_model((28,28),10,x_train,y_train)
```

    [1m1875/1875[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.7716 - loss: 17.6088
    


```python
# evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
```

    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - accuracy: 0.8739 - loss: 6.7864
    
