'''
Data Scientist Jr.: Dr.Eddy Giusepe Chirinos Isidro
Ver um exemplo de código na KAGGLE:
https://www.kaggle.com/utshabkumarghosh/pima-indian-diabetes-ann
 Ver o seguinte link, também:
# https://www.hashtagtreinamentos.com/como-trabalhar-com-arquivos-csv-no-python?gclid=CjwKCAiA78aNBhAlEiwA7B76p6JT6W93G0azb1pRQcFzD1mF-9U-8mInHvtw1own2cMNQSDc-D2evhoCwqsQAvD_BwE
'''
print("#####################################")
print("Importamos as bibliotecas necessárias")
print("#####################################")
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.callbacks
from tensorflow.keras.optimizers import SGD

print("#############################################")
print("Carregando nosso conjunto de Dados - Diabetes")
print("#############################################")
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
dataset_eddy = pd.read_csv("/home/eddygiusepe/5_praticando_Python/Theano_Python_1/diabetes_indians_Kaggle.csv") # Também podemos expressar nossos Dados como D

print("################################################################")
print("Visualização, type e shape de nosso conjunto de Dados - Diabetes")
print("################################################################")
print(dataset_eddy)

print(dataset)
print(type(dataset))
print(dataset.shape)

X = dataset[:,0:8]
Y = dataset[:,8]

print(X)
print(type(X))
print(X.shape)

# Também podemos fazer o seguinte:
x1 = dataset_eddy.iloc[:, 0:8]
y1 = dataset_eddy.iloc[:, 8]
print("Printando o x1: ", x1)
print("")
sc = StandardScaler()
x1 = sc.fit_transform(x1)
print("")
print("Printando o x1: ", x1)
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.25, random_state=1)
# Model:
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(8, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

early_stopping = tensorflow.keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

model.compile(SGD(learning_rate=0.003), "binary_crossentropy", metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
# history = model.fit(X, y, validation_split=0.33, epochs=20, batch_size=10, verbose=0)
history = model.fit(x1_train, y1_train, validation_data=(x1_test, y1_test), epochs=250, batch_size=10, verbose=1, callbacks = [early_stopping])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

_, accuracy = model.evaluate(x1,y1)
print(accuracy)

predictions = model.predict(x1)
for i in range(0, 10):
    print('Predicted: %d, Original: %d' %(predictions[i], y1[i]))




# Define and Compile do nosso Modelo
# model = Sequential()
# model.add(Dense(12, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
# # Fit the model
# model.fit(X, Y, epochs=10, batch_size=10)
# # Evaluate the model
# scores = model.evaluate(X, Y)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))