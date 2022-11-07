#importing modules
import tensorflow as tf
import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#our datasets
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

# #This for loop to check out data
# for i,c in enumerate(celsius_q):
#  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

#Machine Learning model for celsius to Farenhiet
model = tf.keras.Sequential(
    tf.keras.layers.Dense(units=1,input_shape=[1])
)

#compiling our model and minimizing the losses
model.compile(
    loss = 'mean_squared_error',
    optimizer = tf.keras.optimizers.Adam(0.1)
)

#training our model
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

#To see the epoch vs loss graph
# import matplotlib.pyplot as plt
# plt.xlabel('Epoch Number')
# plt.ylabel("Loss Magnitude")
# plt.plot(history.history['loss'])

#Now lets check our moedl's predictions
#print(model.predict([100.0])) gives the answer needed.
# Since it is wrapped in two lists, [0][0] is added.
print("ML's answer for 100.0 celsius is {} farenhiet".format(model.predict([100.0])[0][0]))
