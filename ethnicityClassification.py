import pickle
import  numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
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

X_train = load("ethnicityPickleFiles/X_train.pck")
y_train = load("ethnicityPickleFiles/y_train.pck")
X_test = load("ethnicityPickleFiles/X_test.pck")
y_test = load("ethnicityPickleFiles/y_test.pck")
X_valid = load("ethnicityPickleFiles/X_valid.pck")
y_valid = load("ethnicityPickleFiles/y_valid.pck")
#
# # print("shape xtrain", X_train.shape)
# # print("shape ytrain", y_train.shape)
# # print("shape xtest", X_test.shape)
# # print("shape ytest", y_test.shape)
# # print("shape xvalid", X_valid.shape)
# # print("shape yvalid", y_valid.shape)
#
#


# ################################ Training #########################
# epochs = 12
# batch_size = 16
#
# ethnicityClassificationModel = Sequential([
#     InputLayer(input_shape=(48,48,1)),
#     Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
#     MaxPooling2D((2, 2)),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(rate=0.5),
#     Dense(5, activation='softmax')
# ])
#
# ethnicityClassificationModel.compile(optimizer=opt.Adam(learning_rate=0.0005), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
#
# history = ethnicityClassificationModel.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid), batch_size=batch_size)
#
#
# ########################## plotting training and testing accuracy and loss ####################
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('ethnicity model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('ethnicity model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


#
# ##################### printing the classification report and confusion matrix #########################
# y_pred_test = np.argmax(ethnicityClassificationModel.predict(X_test), axis=1)
# y_pred_train = np.argmax(ethnicityClassificationModel.predict(X_train), axis=1)
# y_true_test = np.argmax(y_test, axis=1)
# y_true_train = np.argmax(y_train, axis=1)
#
# print('Train Classification report')
# print(classification_report(y_pred_train, y_true_train))
#
# print('Test Classification report')
# print(classification_report(y_pred_test, y_true_test))
#
# print('Test Confusion Matrix')
# print(confusion_matrix(y_true_test, y_pred_test))
#
# print('Train Confusion Matrix')
# print(confusion_matrix(y_true_train, y_pred_train))


#
# ####################### validating on X_test #########################
# print(ethnicityClassificationModel.evaluate(X_test, y_test, verbose=1))
#


# ####################### saving and validating again after loading ####################
# ethnicityClassificationModel.save("BestEthnicityModel")
loaded = keras.models.load_model("BestEthnicityModel")
loss1, acc1 = loaded.evaluate(X_test, y_test, verbose=1)
print("ethnicity model Loss on X_test and y_test=", loss1)
print("ethnicity model accuracy on X_test and y_test=", acc1)


##################### confusion matrix of the best model ######################
#
# Test Confusion Matrix
# array([[1797,   75,   49,   98,   28],
#        [  49,  747,    6,   48,    7],
#        [ 123,   52,  476,   17,   15],
#        [ 158,  109,    7,  500,   26],
#        [ 172,   54,   15,   66,   47]])
# Train Confusion Matrix
# array([[5414,  203,   86,  223,   98],
#        [ 101, 2526,   12,  102,   16],
#        [ 252,   98, 1643,   49,   29],
#        [ 354,  261,   25, 1693,   49],
#        [ 496,  169,   38,  161,  125]])