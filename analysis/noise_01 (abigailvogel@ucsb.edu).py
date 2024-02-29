from data_generation.advanced import simulate_RK45
from data_generation.partial import simulate_RK45_partial
from data_generation.orbit_generator import generate_state_vectors

import numpy as np

rng = np.random.default_rng()

initial_state = generate_state_vectors(100)

update_time = 300


planet_selection = np.array([0, 1, 2])
masses = np.load("data/sol_masses.npy")[planet_selection]
planet_state = np.load("data/sol_state_vectors.npy")[planet_selection]

planet_solution = simulate_RK45(
    planet_state,
    update_time,
    masses=masses,
    dt=1,
    lock=0
)

print("planet solution ready")

sat_solution = simulate_RK45_partial(
    initial_state,
    update_time,
    planet_solution,
    dt=1,
    masses=masses
)

print("sat solution ready")


import tensorflow as tf

model = tf.keras.models.load_model(f"saved_models/ANN_06_{update_time}")

noise_amounts = np.linspace(0.0, 1.0, 10, endpoint=False)

print("model ready")

def add_noise(data, amount):
    data = np.copy(data)

    targets = data[:int(len(data) * amount)]

    targets *= rng.normal(1, 0.1, targets.shape)

    rng.shuffle(data)
    return data

def accuracy(y_true, y_pred):
    return np.mean(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=2)), axis=0)

true_output = sat_solution(update_time)

for i, amount in enumerate(noise_amounts):
    altered_initial_state = add_noise(initial_state, amount)

    guess = np.array(model.predict(altered_initial_state))

    print(i, amount, accuracy(true_output, guess))