import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Load IMDB dataset (binary sentiment classification: positive/negative)
max_features = 10000   # number of words to consider
maxlen = 200           # cut texts after 200 words

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to ensure equal length
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Build a simple RNN model for NLP
model = Sequential()
model.add(Embedding(max_features, 32))          # word embeddings
model.add(SimpleRNN(32))                        # recurrent layer
model.add(Dense(1, activation='sigmoid'))       # output layer (binary classification)

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", accuracy)