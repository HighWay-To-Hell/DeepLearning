from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pickle
import numpy as np

data_dir = r'data'

with open(data_dir + '/X_train', 'rb') as f:
    X_train = pickle.load(f)
with open(data_dir + '/Y_train', 'rb') as f:
    Y_train = pickle.load(f)
with open(data_dir + '/X_dev', 'rb') as f:
    X_dev = pickle.load(f)
with open(data_dir + '/Y_dev', 'rb') as f:
    Y_dev = pickle.load(f)

# X 的原shape=(num of samples, num of features = num of timesteps * num of features per timestep)
# 将其reshape成(num of samples, num of timesteps, num of features per timestep)
X_train = np.reshape(X_train, (-1, 99, 13))
X_dev = np.reshape(X_dev, (-1, 99, 13))

model = Sequential()
model.add(LSTM(99, dropout=0.1, recurrent_dropout=0.1, input_shape=(None, 13)))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=50, batch_size=32,
          validation_data=(X_dev, Y_dev))
