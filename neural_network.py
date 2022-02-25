# https://stackoverflow.com/questions/47380267/tensorflow-cannot-install-tensorflow-from-anaconda
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

(x_train, labels_train), (x_test, labels_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(labels_train, 10)
y_test = to_categorical(labels_test, 10)

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation='softmax'))

# compiles the Keras model, gives it a loss functin, the method to optimise the loss function and 
# the metric that will be printed during each epoch
model.compile(loss="categorical_crossentropy", optimizer= 'sgd', metrics=['accuracy'])

print (x_train.shape)
print (x_test.shape)

model.summary()
callbacks = []
# set early stopping criteria
pat = 10 #this is the number of epochs with no improvement after which the training will stop
early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1) # validation loss is monitored


history = model.fit(x_train, y_train, epochs=100, callbacks=[early_stopping], validation_data=(x_test, y_test), batch_size=256)



outputs = model.predict(x_test)
labels_predicted = np.argmax(outputs, axis=1)
misclassified = sum(labels_predicted != labels_test)
print('Percentage misclassified = ', 100 * misclassified / labels_test.size)

# plot_model(model, to_file='network_structure.png', show_shapes=True)
