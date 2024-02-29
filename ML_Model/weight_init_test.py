import numpy as np

dt = 60

weights_1d = [
    np.array(
        [
            [1, dt],
            [-1, -dt],
            [0, 1],
            [0, -1],
        ]
    ),
    np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ),

    np.array(
        [
            [1, -1, 0, 0],
            [0, 0, 1, -1],
        ]
    ),
]

weights_1d = [np.transpose(m) for m in weights_1d]

test_input = np.array([100, -1, 200, -2, 300, -3])

def expand_weight_mat(mat, dims):
    result = np.zeros((mat.shape[0] * dims, mat.shape[1] * dims))
    
    for i in range(dims):
        result[
            mat.shape[0] * i : mat.shape[0] * (i + 1),
            mat.shape[1] * i : mat.shape[1] * (i + 1)
        ] = mat

    #print(result)
    return result

    

#dims = 3
#weights = [expand_weight_mat(w, dims) for w in weights_1d]
#print(weights)
identity = lambda x: x

def test_eval(inputs, weights, act):
    for i, mat in enumerate(weights):
        inputs = act[i](mat @ inputs)
    return inputs

def get_full_weight_list(dims, width, depth):
    result = []
    result.append(np.zeros((width, dims * 2)))
    result[0][:4 * dims] = expand_weight_mat(weights_1d[0], dims)

    for i in range(1, depth - 1):
        result.append(np.zeros((width, width)))
        result[i][:4 * dims, :4 * dims] = expand_weight_mat(weights_1d[2], dims)

    result.append(np.zeros((dims * 2, width)))
    result[-1][:, :4 * dims] = expand_weight_mat(weights_1d[3], dims)

    return result

# for mat in get_full_weight_list(3, 20, 10):
#     print(mat)

def get_layer_weights(dims, width):
    inp_layer_mat = np.zeros((width, dims * 2), dtype=np.float32)
    inp_layer_mat[:4 * dims] = expand_weight_mat(weights_1d[0], dims)

    mid_layer_mat = np.zeros((width, width), dtype=np.float32)
    mid_layer_mat[:4 * dims, :4 * dims] = expand_weight_mat(weights_1d[1], dims)


    out_layer_mat = np.zeros((dims * 2, width), dtype=np.float32)
    out_layer_mat[:, :4 * dims] = expand_weight_mat(weights_1d[2], dims)

    return (
        tf.convert_to_tensor(inp_layer_mat), 
        tf.convert_to_tensor(mid_layer_mat), 
        tf.convert_to_tensor(out_layer_mat)
    )

from data_generation.full_sim import generate_start_stop


# other libraries
import tensorflow as tf
from tensorflow import keras, optimizers
from tensorflow.keras import layers, losses, callbacks, Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# problem parameters
num_sats = int(1e6)  # num satellites to simulate
dims = 3 # how many dimensions
tgt_time = 60 # how far ahead
generate_new_data = False

# model config parameters
mid_layers = 20 # number of hidden layers
mid_layer_width = 256 # width of each hidden layer

# training parameters
train_frac = 0.8 # test-train-split ratio
epochs = 1000
batch_size = 16
vel_priority = 100

# LR parameters
start_alpha = 0.0002
patience = 3
factor = 0.5
min_alpha = 1e-10

# generate data
if generate_new_data:
    all_data = generate_start_stop(num_sats, tgt_time, dims=dims)
    np.save("data/ANN_06_temp.npy", all_data)

else:
    all_data = np.load("data/ANN_06_temp.npy")



x_full = all_data[0]
y_full = all_data[1]


x_train, x_test, y_train, y_test = train_test_split(x_full, y_full)

print("Data generated")

scale_matrix = tf.convert_to_tensor(
    np.array(
        [
            [np.std(all_data[:, :, 0, :]) * dims], 
            [np.std(all_data[:, :, 1, :]) * dims]
        ], 
        dtype=np.float32
    )
)

# does all normalization work
# might be simpler way with preset weights, but this works
class CustomNormalizer(layers.Layer):
    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, inputs):
        return inputs * self.factor

normalizer = CustomNormalizer(1 / scale_matrix)
inv_normalizer = CustomNormalizer(scale_matrix)

inp, mid, out = get_layer_weights(dims, mid_layer_width)
inp_layer = keras.layers.Dense(mid_layer_width, activation="relu")
mid_layers = [keras.layers.Dense(mid_layer_width, activation="relu") for i in range(1, mid_layers)]
out_layer = keras.layers.Dense(2 * dims)

# main model
# a lot of reshaping and normalization surrounding the dense core
model = Sequential(
    [
        layers.InputLayer(input_shape=(2, dims)),
        normalizer,
        layers.Flatten(),
        inp_layer,
        *mid_layers,
        out_layer,
        layers.Reshape((2, dims)),
        inv_normalizer
    ]
)

model(np.array([[[0, 0, 0], [0, 0, 0]]], dtype=np.float32))

inp_layer.set_weights([inp, inp_layer.get_weights()[1]])
for l in mid_layers:
    l.set_weights([mid, l.get_weights()[1]])
out_layer.set_weights([out, out_layer.get_weights()[1]])

# printed-out accuracy
def accuracy(y_true, y_pred):
    return K.mean(K.sqrt(K.sum((y_true - y_pred) ** 2, axis=2)), axis=0)


print(accuracy(y_full, reference_model(x_full)))


# so that the model prints out seperate accuracies for position and velocity
class CustomCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("", epoch,[float(x) for x in [
            *accuracy(y_train[:1000], self.model(x_train[:1000])), 
            *accuracy(y_test[:1000], self.model(x_test[:1000]))
        ]])


# prioritize velocity, via magnification
loss_scale = tf.convert_to_tensor(np.array([[[1] * dims, [vel_priority] * dims]], dtype=np.float32))


def custom_loss(y_true, y_pred):
    return losses.mean_squared_error(
        y_true * loss_scale, 
        y_pred * loss_scale
    )

model.compile(
    optimizer=optimizers.Adam(learning_rate=start_alpha),
    loss=custom_loss
)

print("Model ready...")

print(model.summary())

print("Training begins!")

model.fit(
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
