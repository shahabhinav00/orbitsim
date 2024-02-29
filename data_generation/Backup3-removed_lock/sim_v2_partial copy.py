from sim_v2 import get_accel_vectors, GRAV_CONSTANT
from sim_v2_advanced import OutputReshaper
import numpy as np
import scipy as sp

try:
    from tqdm import tqdm

except ModuleNotFoundError:

    def tqdm(x):
        return x


# This module is for simulators that are given
# the pre-computed states of the solar system
# by simulating only the satellites, we gain a lot of speed


def get_pull_on_single_object(distance_vectors, masses=None):
    num_planets = len(distance_vectors)

    if masses is None:
        masses = np.ones(num_planets)

    dist_squareds = np.sum(distance_vectors**2, axis=1)

    ignore = dist_squareds == 0
    dist_squareds[ignore] = 1

    #print(distance_vectors.shape)
    #print(dist_squareds.shape)

    pulls = masses * GRAV_CONSTANT / dist_squareds
    pull_vectors = pulls[..., None] * (
        distance_vectors / np.sqrt(dist_squareds)[..., None]
    )

    pull_vectors[ignore] = 0

    return np.sum(pull_vectors, axis=0)


# If you have the solar system trajectories pre-computed, use this
# should be quite a bit faster


def get_accel_vectors_partial(mass_positions, tgt_positions, masses=None):
    if masses is None:
        masses = np.ones(len(mass_positions))

    mass_idx, tgt_idx = np.meshgrid(
        np.arange(len(mass_positions)), np.arange(len(tgt_positions)), indexing="ij"
    )

    distance_vectors = mass_positions[mass_idx] - tgt_positions[tgt_idx]
    dist_squareds = np.sum(distance_vectors**2, axis=2)

    pulls = masses[mass_idx] * GRAV_CONSTANT / dist_squareds

    pull_vectors = pulls[..., None] * (
        distance_vectors / np.sqrt(dist_squareds)[..., None]
    )

    return np.sum(pull_vectors, axis=0)


def simulate_RK45_partial(state, time, sol, masses=None, lock=None):
    shape = state.shape

    def sim_func(t, flat):
        mass_state = sol(t)
        tgt_state = np.reshape(flat, shape)

        result = np.empty_like(tgt_state)
        result[:, 0] = tgt_state[:, 1]
        result[:, 1] = get_accel_vectors_partial(
            mass_state[:, 0], tgt_state[:, 0], masses=masses
        )

        if lock is not None:
            result[:, 0] -= mass_state[lock, 1]
            result[:, 1] -= get_pull_on_single_object(
                mass_state[lock, 0] - mass_state[:, 0], masses=masses
            )

        return result.flatten()

    result = sp.integrate.solve_ivp(
        sim_func,
        (0, time),
        state.flatten(),
        dense_output=True,
        method="RK45",
    )

    solution = result.sol

    return OutputReshaper(solution, shape)


def simulate_EUL_partial(state, time, sol, dt=3600, masses=None, lock=None):
    num_steps = time // dt

    result = np.empty((num_steps, *state.shape))

    result[0] = state

    if lock is not None:
        result[0] -= result[0, lock]

    for i in tqdm(range(1, num_steps)):
        t = i * dt
        mass_state = sol(t)[:, 0]

        accel_vectors = get_accel_vectors_partial(
            mass_state, result[i - 1, :, 0], masses=masses
        )

        if lock is not None:
            accel_vectors -= get_pull_on_single_object(
                mass_state - mass_state[lock], masses=masses
            )

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

class combine_solutions:
    def __init__(self, *solutions):
        self.solutions = solutions

    def __call__(self, t):
        return np.concatenate([sol(t) for sol in self.solutions])