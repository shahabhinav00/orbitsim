
# importing data generator
from data_generation.full_sim import generate
from noise_utils import add_noise
from load_utils import norm, inv_norm, custom_loss


# other libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras, optimizers
from tensorflow.keras import layers, losses, callbacks, Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import sys

# problem parameters
num_sats = 1000000  # num satellites to simulate
dims = 3  # how many dimensions
tgt_time = -2  # how far ahead
generate_new_data = True

noise = 0.001

# model config parameters
mid_layers = 4 # number of hidden layers
mid_layer_width = 256 # width of each hidden layer

# training parameters
train_frac = 0.8 # test-train-split ratio
epochs = 50 # enough, don't change
batch_size = 8
vel_priority = 100

# LR parameters
start_alpha = 0.0001
patience = 2
factor = 0.5
min_alpha = 1e-9

rng = np.random.default_rng()

if tgt_time == -1:
    tgt_time = int(input("Prediction target time: "))

elif tgt_time == -2:
    tgt_time = int(sys.argv[1])

# generate data
if generate_new_data:
    all_data = generate(num_sats, tgt_time, dims=dims)
    np.save("data/ANN_06_temp.npy", all_data)

else:
    all_data = np.load("data/ANN_06_temp.npy")

main_filepath = f"saved_models/ANN_07_{tgt_time}D"


x_full = add_noise(all_data[0], noise, noise)
y_full = add_noise(all_data[1], noise, noise)


x_train, x_test, y_train, y_test = train_test_split(x_full, y_full)

print("Data generated")

# main model
# a lot of reshaping and normalization surrounding the dense core
model = Sequential(
    [
        layers.InputLayer(input_shape=(2, dims)),
        
        layers.Flatten(),
        norm,
        *[layers.Dense(mid_layer_width, activation="relu") for i in range(mid_layers)],
        layers.Dense(2 * dims),
        inv_norm,
        layers.Reshape((2, dims)),
        
    ]
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=start_alpha),
    loss=custom_loss
)

print("Model ready...")

print(model.summary())
print("Training begins!")

error_log = []

# printed-out accuracy
def accuracy(y_true, y_pred):
    return K.mean(K.sqrt(K.sum((y_true - y_pred) ** 2, axis=2)), axis=0)


# so that the model prints out seperate accuracies for position and velocity
class CustomCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        result = [tgt_time, epoch, *[float(x) for x in [
                    *accuracy(y_train[:1000], self.model.predict(x_train[:1000])), 
                    *accuracy(y_test[:1000], self.model.predict(x_test[:1000])),
                ]]]
        tf.print("", *result)
        error_log.append(str(result))

result = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[
        CustomCallback(),
        callbacks.ReduceLROnPlateau(
            min_lr=min_alpha,
            factor=factor,
            patience=patience
        ),
    ],
    verbose=1
)

model.save(main_filepath)

print(result.history)

with open(main_filepath + '/history.txt', "w") as file:
    file.write(str(result.history))

with open(main_filepath + '/log.txt', "w") as file:
    for line in error_log:
        file.write(str(line) + "\n")