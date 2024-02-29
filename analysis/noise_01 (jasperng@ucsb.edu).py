from data_generation.advanced import simulate_RK45
from data_generation.partial import simulate_RK45_partial
from data_generation.full_sim import generate
from load_utils import load_model
import numpy as np
import tensorflow as tf

rng = np.random.default_rng()


update_time = 60

data = generate(100, update_time)


model = load_model(f"saved_models/ANN_06_{update_time}_FINAL")


noise_amounts = np.linspace(0.0, 0.01, 11)

print("model ready")


def add_noise(data, amount):
    data = np.copy(data)

    targets = data[:int(len(data) * amount)]

    targets *= rng.normal(1, 0.1, targets.shape)

    return data


def accuracy(y_true, y_pred):
    return np.mean(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=2)), axis=0)


for i, amount in enumerate(noise_amounts):
    inp = data[0] * rng.normal(1, amount, data[0].shape)
    guess = np.array(model.predict(inp, verbose=0))
    print(i, amount)
    print(accuracy(data[0], inp))
    print(accuracy(data[1], guess))

    print()