# -*- coding: utf-8 -*-

!wget http://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz
!tar -xvzf WISDM_ar_latest.tar.gz
!rm -rf WISDM_ar_latest.tar.gz

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import keras
from keras import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import LSTM, TimeDistributed, Activation, Dense, RepeatVector
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split
import re

# %matplotlib inline

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42
columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
df = pd.read_csv('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt', header = None, names = columns)
if re.search(';', df.iloc[0, 5]) != None :
    df['z-axis'] = df['z-axis'].map(lambda x : float(re.sub(';', '', str(x))))
df = df.dropna()
df.head()

N_TIME_STEPS = 200
N_FEATURES = 3
step = 20
labels = []
for i in range(0, len(df) - N_TIME_STEPS, step):
    label = stats.mode(df['activity'][i: i + N_TIME_STEPS])[0][0]
    labels.append(label)

"""## LSTM Timeseries 전처리"""

data_gen = TimeseriesGenerator(np.array(df[['x-axis', 'y-axis', 'z-axis']]), np.repeat(0, len(df)),
                               length=200, sampling_rate=1, 
                               stride = 20, batch_size = len(labels))

x_all = data_gen[0][0]

x_all[0].shape

y_all = np.asarray(pd.get_dummies(labels), dtype = np.float32)

pd.DataFrame(labels).groupby(0).size()

temp_idx = 5000
print(labels[temp_idx])
y_all[temp_idx]

# train, test 로 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=42)

assert X_train.shape[0] == len(y_train)



y_train[0]

idx = 0

plt.figure(1)
plt.subplot(311)
plt.plot(X_train[idx, :, 0], 'o-')
plt.legend('x', loc = 'upper right')
plt.title(labels[idx])

plt.subplot(312)
plt.plot(X_train[idx, :, 1], 'or-')
plt.legend('y', loc = 'upper right')

plt.subplot(313)
plt.plot(X_train[idx, :, 2], 'og-')
plt.legend('z', loc = 'upper right')

plt.show()

"""### 1. LSTM만으로 예측"""

hidden_size = 50
output_dim = len(np.unique(labels))

model = Sequential()
model.add(LSTM(hidden_size, input_shape = (N_TIME_STEPS, N_FEATURES)))
# model.add(LSTM(hidden_size, return_sequences=True))
# model.add(Dropout(0.5))
# model.add(TimeDistributed(Dense(output_dim)))
model.add(Dense(output_dim))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 10, batch_size = 1024)

hist.history.keys()

# 5. 모델 학습 과정 표시하기
# %matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

"""### 2. DNN"""

# train, test 로 데이터 나누기
y_DNNall = np.asarray(pd.get_dummies(df['activity']), dtype = np.float32)
X_DNNtrain, X_DNNtest, y_DNNtrain, y_DNNtest = train_test_split(df[['x-axis', 'y-axis', 'z-axis']], y_DNNall, test_size=0.3, random_state=42)

hidden_size = 50
output_dim = len(np.unique(df['activity']))

model = Sequential()
model.add(Dense(hidden_size, activation = 'relu', input_dim = N_FEATURES))
model.add(Dense(int(hidden_size/2), activation = 'relu'))
model.add(Dense(int(hidden_size/4), activation = 'relu'))
model.add(Dense(output_dim))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_DNNtrain, y_DNNtrain, validation_data=(X_DNNtest, y_DNNtest), epochs = 10, batch_size = 256)

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# %matplotlib inline

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

"""### 3. stacked LSTM"""

hidden_size = 50
output_dim = len(np.unique(labels))

model = Sequential()
model.add(LSTM(hidden_size, return_sequences=True, input_shape = (N_TIME_STEPS, N_FEATURES)))
model.add(LSTM(hidden_size))
# model.add(Dropout(0.5))
# model.add(TimeDistributed(Dense(output_dim)))
model.add(Dense(output_dim))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 10, batch_size = 1024)



model = Sequential()
model.add(LSTM(hidden_size, return_sequences=True, input_shape = (N_TIME_STEPS, N_FEATURES)))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size))
# model.add(Dropout(0.5))
# model.add(TimeDistributed(Dense(output_dim)))
model.add(Dense(output_dim))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 10, batch_size = 1024)



"""### 4. stacked LSTM + Autoencoder"""

new_df = df[np.isin(df['activity'], ['Walking', 'Sitting'])].reset_index(drop=True)

df_walking = df[df['activity'] == 'Walking'].reset_index(drop=True)

df_sitting = df[df['activity'] == 'Sitting'].reset_index(drop=True)

print(df_walking.shape)
df_sitting.shape

N_TIME_STEPS = 200
N_FEATURES = 3
step = 20
labels2 = []
for i in range(0, len(df_walking) - N_TIME_STEPS, step):
    label = stats.mode(df_walking['activity'][i: i + N_TIME_STEPS])[0][0]
    labels2.append(label)

df_sitting.head()

idx = 0

plt.figure(1)
plt.subplot(311)
plt.plot(df_walking[df_walking['user'] == 33].iloc[:, 3], 'o-')
plt.legend('x', loc = 'upper right')

plt.figure(1)
plt.subplot(312)
plt.plot(df_walking[df_walking['user'] == 33].iloc[:, 4], 'o-')
plt.legend('y', loc = 'upper right')

plt.figure(1)
plt.subplot(313)
plt.plot(df_walking[df_walking['user'] == 33].iloc[:, 5], 'o-')
plt.legend('z', loc = 'upper right')


plt.show()

plt.figure(1)
plt.subplot(311)
plt.plot(df_sitting[df_sitting['user'] == 33].iloc[:, 3], 'o-')
plt.legend('x', loc = 'upper right')

plt.figure(1)
plt.subplot(312)
plt.plot(df_sitting[df_sitting['user'] == 33].iloc[:, 4], 'o-')
plt.legend('y', loc = 'upper right')

plt.figure(1)
plt.subplot(313)
plt.plot(df_sitting[df_sitting['user'] == 33].iloc[:, 5], 'o-')
plt.legend('z', loc = 'upper right')


plt.show()



data_gen = TimeseriesGenerator(np.array(df_walking[['x-axis', 'y-axis', 'z-axis']]), np.repeat(0, len(df_walking)),
                               length=200, sampling_rate=1, 
                               stride = 20, batch_size = len(labels2))

x_all = data_gen[0][0]

y_all = np.asarray(pd.get_dummies(labels2), dtype = np.float32)

# train, test 로 데이터 나누기
X_train, X_test = train_test_split(x_all, test_size=0.3, random_state=42)

idx = 0

plt.figure(1)
plt.subplot(311)
plt.plot(X_train[idx, :, 0], 'o-')
plt.legend('x', loc = 'upper right')
plt.title(labels2[idx])

plt.subplot(312)
plt.plot(X_train[idx, :, 1], 'or-')
plt.legend('y', loc = 'upper right')

plt.subplot(313)
plt.plot(X_train[idx, :, 2], 'og-')
plt.legend('z', loc = 'upper right')

plt.show()





# define model
model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(N_TIME_STEPS,N_FEATURES)))
model.add(RepeatVector(N_TIME_STEPS))
model.add(LSTM(50, activation='tanh', return_sequences=True))
model.add(TimeDistributed(Dense(3)))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=10, batch_size= 256)

model.layers[0].output

yhat.shape

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# %matplotlib inline

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

from numpy import array
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

input1 = Input(shape=(N_TIME_STEPS,N_FEATURES))

encoded1 = LSTM(50, activation='tanh')(input1)
encoded1 = RepeatVector(N_TIME_STEPS)(encoded1)
encoded2 = LSTM(50, activation='tanh', return_sequences=True)(encoded1)
decoded1 = TimeDistributed(Dense(3))(encoded2)
autoencoder = Model(inputs = input1, outputs = decoded1)

encoder = Model(input1, encoded2)
encoded_input = Input(shape = (200, 50))

autoencoder.layers[-1].input_shape

decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=10, batch_size= 256)

X_test_hat = encoder.predict(X_test)

data_gen = TimeseriesGenerator(np.array(df_sitting[['x-axis', 'y-axis', 'z-axis']]), np.repeat(0, len(df_sitting)),
                               length=200, sampling_rate=1, 
                               stride = 20, batch_size = len(labels2))

x_all = data_gen[0][0]

X_sitting_hat = encoder.predict(x_all)

X_train_hat = encoder.predict(X_train)

X_test_hat.shape

X_sitting_hat.shape

X_all = np.concatenate([X_train_hat, X_test_hat, X_sitting_hat], axis = 0)

X_all.shape

y_all = np.concatenate([np.repeat(0, X_all.shape[0]-X_sitting_hat.shape[0]), np.repeat(1, X_sitting_hat.shape[0])], axis = 0)

y_all.shape

y_all = np.reshape(y_all, (24197, 1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42)

model = Sequential()
model.add(Dense(25, activation='relu', input_shape = (200, 50)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# %matplotlib inline

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))