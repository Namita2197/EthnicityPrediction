import pickle

from matplotlib import pyplot as plt
from tensorflow import *
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,InputLayer, Dropout, BatchNormalization, Flatten, Dense, MaxPooling2D
from tensorflow.keras.models import Sequential
import tensorflow.keras.optimizers as opt


def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)

X_train = load("genderPickleFiles/X_train.pck")
y_train = load("genderPickleFiles/y_train.pck")
X_test = load("genderPickleFiles/X_test.pck")
y_test = load("genderPickleFiles/y_test.pck")
X_valid = load("genderPickleFiles/X_valid.pck")
y_valid = load("genderPickleFiles/y_valid.pck")
#
# # printing the shapes of all the loaded files to check if they are correct
# # print("shape xtrain", X_train.shape)
# # print("shape ytrain", y_train.shape)
# # print("shape xtest", X_test.shape)
# # print("shape ytest", y_test.shape)
# # print("shape xvalid", X_valid.shape)
# # print("shape yvalid", y_valid.shape)
#
#

# ################################ Training #########################
# epochs = 20
# batch_size = 16
#
# genderClassificationModel = Sequential([
#     InputLayer(input_shape=(48,48,1)),
#     Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
#     MaxPooling2D((2, 2)),
#     BatchNormalization(),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Dropout(rate=0.6),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dropout(rate=0.5),
#     Dense(2, activation='softmax')
# ])
#
# genderClassificationModel.compile(optimizer=opt.Adam(learning_rate=0.0001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
#
# history = genderClassificationModel.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid), batch_size=batch_size)
#


# ########################## plotting training and testing accuracy and loss ####################
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('gender model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('gender model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#


# ####################### validating on X_test #########################
# print(genderClassificationModel.evaluate(X_test, y_test, verbose=1))
#


# ####################### saving and validating again after loading ####################
# genderClassificationModel.save("BestGenderModel")
loaded = keras.models.load_model("BestGenderModel")
finalLoss, finalAcc = loaded.evaluate(X_test, y_test, verbose=1)
print("gender model Loss on X_test and y_test=", finalLoss)
print("gender model accuracy on X_test and y_test=", finalAcc)