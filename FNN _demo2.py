import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# One-hot encode target labels
y = to_categorical(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build Feedforward Neural Network (FNN)
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))   # hidden layer 1
model.add(Dense(12, activation='relu'))                # hidden layer 2
model.add(Dense(3, activation='softmax'))              # output layer (3 classes)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", accuracy)