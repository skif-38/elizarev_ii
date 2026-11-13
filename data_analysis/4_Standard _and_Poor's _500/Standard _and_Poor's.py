import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

data=pd.read_csv("data.csv")
y=data["SP500"].values
y1 = np.convolve(y, [1/100]*100)

K=20
for i in range(K):
    if i>0:
        X=np.concatenate((X, y1[i:-K+i, np.newaxis]), axis=-1)
    else:
        X=y1[i:-K+i, np.newaxis].copy()
Y=y1[K:]
print(X.shape, Y.shape)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(K,)))
model.add(tf.keras.layers.Dense(1,activation='linear', kernel_regularizer='l1', bias_regularizer='l1'))
tf.keras.utils.plot_model(model,
show_layer_activations=True,show_shapes=True, to_file='model.png')
from sklearn.model_selection import train_test_split
xtr, xtest, ytr, ytest = train_test_split(X, Y)
model.compile(optimizer='Adam', loss='mse')
es=tf.keras.callbacks.EarlyStopping(patience=5, mode='min', monitor='val_loss')
model.fit(xtr, ytr, epochs=1000, validation_split=0.2, callbacks=[es])
yp=model.predict(xtest)

plt.figure()
plt.plot(ytest)
plt.plot(yp)
plt.legend()

plt.figure()
plt.scatter(ytest, yp)
plt.scatter(ytest, ytest);