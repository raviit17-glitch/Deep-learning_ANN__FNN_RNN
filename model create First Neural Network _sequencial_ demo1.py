
#model create
from tensorflow.keras.models import Sequential

# model lyer(input, hidden,out)
from tensorflow.keras.layers import Dense
#model create function
model = Sequential()

#model layer with 
#weight instaillise
# add activation_function="relu"
layers_hidden_input=Dense(16,input_shape=(8,), kernel_initializer="random_uniform", activation="relu");


#model hidden layer add(1)
model.add(layers_hidden_input);

#complie model
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"]);   

#Model Traing
model.fit(X  , y, epochs=100, batch_size=32);  

#Model predit       
y_pred=model.predict(X_test);