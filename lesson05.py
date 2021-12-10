'''
Data Scientist Jr.: Dr.Eddy Giusepe Chirinos Isidro
Ver também um exemplo de código na KAGGLE:
https://www.kaggle.com/utshabkumarghosh/pima-indian-diabetes-ann
'''
# Importamos  as nossas bibliotecas
# Ver o seguinte link:
# https://www.hashtagtreinamentos.com/como-trabalhar-com-arquivos-csv-no-python?gclid=CjwKCAiA78aNBhAlEiwA7B76p6JT6W93G0azb1pRQcFzD1mF-9U-8mInHvtw1own2cMNQSDc-D2evhoCwqsQAvD_BwE
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

# Carregar o conjunto de dados
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
dataset_eddy = pd.read_csv("/home/eddygiusepe/5_praticando_Python/Theano_Python_1/diabetes_indians_Kaggle.csv") # Também podemos expressar nossos Dados como D
print(dataset_eddy)

print(dataset)
print(type(dataset))
print(dataset.shape)

X = dataset[:,0:8]
Y = dataset[:,8]

print(X)
print(type(X))
print(X.shape)


# Define and Compile do nosso Modelo
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)
# Evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))