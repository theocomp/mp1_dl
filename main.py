import mp1
import numpy as np
import matplotlib as mlp
mlp.use("TkAgg")
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGE_SIZE = 100

def generate_train_test(nb_sample, noise, bool):
    [X_train, Y_train] = mp1.generate_dataset_classification(nb_sample, noise, bool)
    [X_test, Y_test] = mp1.generate_dataset_classification(nb_sample, noise, bool)
    return([X_train, Y_train], [X_test, Y_test])


def generate_train_test_reg(nb_sample, noise):
    [X_train, Y_train] = mp1.generate_dataset_regression(nb_sample, noise)
    [X_test, Y_test] = mp1.generate_dataset_regression(nb_sample, noise)
    return([X_train, Y_train], [X_test, Y_test])

#[X_trainr, Y_trainr], [X_testr, Y_testr] = generate_train_test_reg(1000, 20)
#[X_train, Y_train2], [X_test2, Y_test2] = generate_train_test(1000, 20, True)

def save_data():
    np.save('X_train', X_train)
    np.save('X_test', X_test)
    np.save('Y_train', Y_train)
    np.save('Y_test', Y_test)
    np.save('X_train2', X_train2)
    np.save('X_test2', X_test2)
    np.save('Y_train2', Y_train2)
    np.save('Y_test2', Y_test2)
    np.save('X_trainr', X_trainr)
    np.save('X_testr', X_testr)
    np.save('Y_trainr', Y_trainr)
    np.save('Y_testr', Y_testr)


def load_data():
    X_train = np.load("/Users/theocomperot/PycharmProjects/mp1_dl/X_train.npy")
    X_test = np.load("/Users/theocomperot/PycharmProjects/mp1_dl/X_test.npy")
    Y_train = np.load("/Users/theocomperot/PycharmProjects/mp1_dl/Y_train.npy")
    Y_test = np.load("/Users/theocomperot/PycharmProjects/mp1_dl/Y_test.npy")
    X_train2 = np.load("/Users/theocomperot/PycharmProjects/mp1_dl/X_train2.npy")
    X_test2 = np.load("/Users/theocomperot/PycharmProjects/mp1_dl/X_test2.npy")
    Y_train2 = np.load("/Users/theocomperot/PycharmProjects/mp1_dl/Y_train2.npy")
    Y_test2 = np.load("/Users/theocomperot/PycharmProjects/mp1_dl/Y_test2.npy")
    X_trainr = np.load("/Users/theocomperot/PycharmProjects/mp1_dl/X_trainr.npy")
    X_testr = np.load("/Users/theocomperot/PycharmProjects/mp1_dl/X_testr.npy")
    Y_trainr = np.load("/Users/theocomperot/PycharmProjects/mp1_dl/Y_trainr.npy")
    Y_testr = np.load("/Users/theocomperot/PycharmProjects/mp1_dl/Y_testr.npy")
    return(X_train, X_test, Y_train, Y_test, X_train2, X_test2, Y_train2, Y_test2, X_trainr, X_testr, Y_trainr, Y_testr)


X_train, X_test, Y_train, Y_test, X_train2, X_test2, Y_train2, Y_test2,X_trainr, X_testr, Y_trainr, Y_testr = load_data()

################
### Q3 : MLP ###
################

# Model

model = Sequential()

model.add(layer=Dense(units=20, input_shape=(10000, ), activation='relu'))
model.add(Dropout(0.25))
model.add(layer=Dense(units=20, input_shape=(20, ), activation='relu'))
model.add(Dropout(0.25))
model.add(layer=Dense(units=3, activation='softmax'))  # 3 categories

# Opti

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit

history = model.fit(X_train, Y_train, epochs=500, batch_size=64)

plt.plot(history.history['loss'])

# CHECK : cat1 = rect, cat2 = disque, cat3 = triangle#

Y_pred = model.predict(X_test)
Y_pred = np.argmax(Y_pred, axis=1)
Y_test = np.argmax(Y_test, axis=1)
bools =(Y_pred != Y_test)

model.evaluate(X_test, Y_test)

# On obtient 100% de précision

##########################
### Q4 : Visualization ###
##########################

weights = model.get_weights()
matrix_weights = np.dot(weights[0], np.dot(weights[2],  weights[4]))

im1 = matrix_weights[:, 0].reshape(100, 100)
im2 = matrix_weights[:, 1].reshape(100, 100)
im3 = matrix_weights[:, 2].reshape(100, 100)
plt.imshow(im1)
plt.show()

################
### Q5 : CNN ###
################

# Train data

X_train2 = X_train2.reshape((1000, 100, 100, 1))
Y_train2 = to_categorical(Y_train2)

# Model

model = Sequential()

model.add(layer=Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))  # max pooling
model.add(Dropout(0.25))
model.add(layer=Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50)))
model.add(MaxPooling2D(pool_size=(2, 2)))  # max pooling
model.add(Dropout(0.25))
model.add(layer=Conv2D(32, (3, 3), activation='relu', input_shape=(25, 25)))
model.add(MaxPooling2D(pool_size=(2, 2)))  # max pooling
model.add(Dropout(0.25))
model.add(Flatten())  # On flatten pour les couches denses d'après
model.add(Dense(64, activation='relu'))  # Couche dense activation relu
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))  # Couche dense activation relu
model.add(Dropout(0.25))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))   # Output layer, 3 catégories

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train2, Y_train2, batch_size=32, epochs=50, verbose=1)


model.evaluate(X_test2, Y_test2)

# We obtain 94% of precision

#######################
### Q6 : Regression ###
#######################


X_trainr = X_trainr.reshape((1000, 100, 100, 1))

# Model

model = Sequential()

model.add(layer=Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))  # max pooling
model.add(Dropout(0.25))
model.add(layer=Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50)))
model.add(MaxPooling2D(pool_size=(2, 2)))  # max pooling
model.add(Dropout(0.25))
model.add(layer=Conv2D(32, (3, 3), activation='relu', input_shape=(25, 25)))
model.add(MaxPooling2D(pool_size=(2, 2)))  # max pooling
model.add(Dropout(0.25))
model.add(Flatten())  # On flatten pour les couches denses d'après
model.add(Dense(64, activation='relu'))  # Couche dense activation relu
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))  # Couche dense activation relu
model.add(Dropout(0.25))
model.add(Dense(6))   # Output layer, 3 catégories

sgd = SGD(lr=0.0001,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_trainr, Y_trainr, batch_size=32, epochs=10)


X_testr = X_testr.reshape((1000, 100, 100, 1))
model.evaluate(X_testr, Y_testr)