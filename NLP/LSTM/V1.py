# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:10:17 2023

@author: Anirudh
"""

# Import Packages.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dropout, Bidirectional, Dense, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

import wikipedia

# Defining some Variables.
epoch_size = 12
batch_size = 32

# Getting the dataset.
print("********************************************")
print("Preparing the datasets")
wikipedia.set_lang('en')
result = wikipedia.summary('Python (Programming Language)')
result += wikipedia.summary('Java (Language)')
result += wikipedia.summary('C (Programming Language)')
result += wikipedia.summary('C++ (Programming Language)')
result += wikipedia.summary('HTML (Language)')
result += wikipedia.summary('CSS (Language)')
result = result.split('.')

divide = int( 0.85 * len(result))

train_data = result[:divide]
test_data = result[divide:]

print("Datasets are sucessfully integrated.")
# Creating Tokenizer dictionary
tokenizer = Tokenizer(num_words = 1000, oov_token = '<oov>')
tokenizer.fit_on_texts(train_data)

# Converting the text into sequence of integers
train_seq = tokenizer.texts_to_sequences(train_data)
test_seq = tokenizer.texts_to_sequences(test_data)

# Converting the sequence to equal sizes by padding
train_pad = pad_sequences(train_seq, maxlen = 1000, padding = "post", truncating = "post")
test_pad = pad_sequences(test_seq, maxlen = 1000, padding = "post", truncating = "post")

# One-hot-encoding
train_labels = to_categorical(train_pad, num_classes = 1000)
test_labels = to_categorical(test_pad, num_classes = 1000)

# Building the model.
model = Sequential()
model.add(Embedding(input_dim = len(tokenizer.word_index) + 1, output_dim = 1000, input_length = 1000))
model.add(LSTM(units = 64, dropout = 0.2, return_sequences = True))
model.add(Dense(1000, activation = 'softmax'))

# Compile the model.
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(train_pad, train_labels, batch_size= batch_size, epochs= epoch_size, validation_split= 0.2)


# Predictions.
def predictor(text):
    generated_text = ""
    for i in range(100):    # Generate 100 next words
        # Tokenize the text
        tokens = tokenizer.texts_to_sequences([text + generated_text])
        

        # Pad the tokens
        tokens_padded = pad_sequences(tokens, maxlen=1000, padding='post')
        #tokens_padded = to_categorical(tokens_padded.flatten, num_classes = 1000)
        
        # Get the predicted index
        output = model.predict(tokens_padded)
        
        predicted_index = np.argmax(output,axis = 1)

        # Convert the index to word
        predicted_word = tokenizer.index_word[predicted_index[0][0]]
    
        # Add the predicted word to the text
        generated_text += ' ' + predicted_word
    
    print("The Generated word:-\n", text + "\n" + generated_text)
    
predictor("HTML with")
