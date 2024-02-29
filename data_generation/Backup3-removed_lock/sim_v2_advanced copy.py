import numpy as np
import scipy as sp
from sim_v2 import get_accel_vectors

try:
    from tqdm import tqdm

except ModuleNotFoundError:

    def tqdm(x):
        return x


# solution to not being able to pickle lambdas
class OutputReshaper:
    def __init__(self, func, shape):
        self.func = func
        self.shape = shape

    def __call__(self, *args):
        return np.reshape(self.func(*args), self.shape)


# Runga-Kutta advanced simulation
# note: dt is maximum dt, use it to ensure accuracy
def simulate_RK45(state, time, masses=None, lock=None, dt=3600):
    state = np.copy(state)

    if lock is not None:
        state -= state[lock]

    shape = state.shape

    def sim_func(t, flat):
        state = np.reshape(flat, shape)
        # result = np.empty_like(state)
        # result[:, 0] = state[:, 1]
        # result[:, 1] = get_accel_vectors(state[:, 0], masses=masses, lock=lock)
        return np.concatenate(state[:, 1], get_accel_vectors(state[:, 0], masses=masses, lock=lock))

    result = sp.integrate.solve_ivp(
        sim_func,
        (0, time),
        state.flatten(),
        dense_output=True,
        method="RK45",
        max_step=dt,
    )

    return OutputReshaper(result.sol, shape)


# simulation via euler's method
def simulate_EUL(state, time, masses=None, lock=None, dt=60):
    num_steps = time // dt

    result = np.empty((num_steps, *state.shape))

    result[0] = state

    if lock is not None:
        result[0] -= result[0, lock]

    for i in tqdm(range(1, num_steps)):
        accel_vectors = get_accel_vectors(result[i - 1, :, 0], masses=masses, lock=lock)
        result[i, :, 1] = result[i - 1, :, 1] + accel_vectors * dt
        result[i, :, 0] += result[i - 1, :, 0] + result[i, :, 1] * dt

    return sp.interpolate.interp1d(
        np.arange(num_steps) * dt,
        result,
        axis=0,
        copy=False,
        assume_sorted=True,
        fill_value="extrapolate",
    )
