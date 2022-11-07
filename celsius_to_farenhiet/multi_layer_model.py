#importing modules
import tensorflow as tf
import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#our datasets
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

#ML model with 3 layers
model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=4, input_shape=[1]),
  tf.keras.layers.Dense(units=4),
  tf.keras.layers.Dense(units=1)
  ])

#compiling our model
model.compile(
    loss='mean_squared_error', 
    optimizer=tf.keras.optimizers.Adam(0.1)
    )

#Training our model
model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")
#Now lets check our moedl's predictions
#print(model.predict([100.0])) gives the answer needed.
# Since it is wrapped in two lists, [0][0] is added.
print("ML's answer for 100.0 celsius is {} farenhiet".format(model.predict([100.0])[0][0]))

#To know about the weights.
# print("These are the l0 variables: {}".format(l0.get_weights()))
# print("These are the l1 variables: {}".format(l1.get_weights()))
# print("These are the l2 variables: {}".format(l2.get_weights()))
