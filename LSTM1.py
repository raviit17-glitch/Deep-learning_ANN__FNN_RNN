import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Example: sequences of length 5, predicting sum parity (even/odd)
X = np.random.randint(0, 10, (1000, 5))   # 1000 sequences, each of length 5
y = np.sum(X, axis=1) % 2                 # target: 0 if sum even, 1 if odd

# Reshape input for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# One-hot encode labels
y = to_categorical(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(32, input_shape=(5, 1)))          # LSTM layer
model.add(Dense(16, activation='relu'))          # hidden layer
model.add(Dense(2, activation='softmax'))        # output layer (2 classes)

# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", accuracy)