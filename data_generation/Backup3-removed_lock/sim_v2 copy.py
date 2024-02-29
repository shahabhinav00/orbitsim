import numpy as np

# km^3 / kg / s^2
GRAV_CONSTANT = 6.674 * (10**-20)

# main simulator
# things are measured in kilometers, kilograms, seconds
# the lock parameter locks an object to the origin,
# as the simulation happens around it
# only function that really matters here is get_accel_vectors


def move(state, dt, accel_vectors):
    state[:, 1] += dt * accel_vectors
    state[:, 0] += dt * state[:, 1]


def get_accel_vectors(positions, masses=None, lock=None):
    num_planets = len(positions)

    if masses is None:
        masses = np.ones(num_planets)

    each = np.arange(num_planets)

    # get distances all at once
    p1s, p2s = np.meshgrid(each, each, indexing="ij")

    # distance_vectors[i, j] = the vector from planet i to planet j
    distance_vectors = positions[p2s] - positions[p1s]

    # dist_squareds[i, j] = the square of the distance from planet i to planet j
    dist_squareds = np.sum(distance_vectors**2, axis=2)

    ignore = (
        dist_squareds == 0
    )  # things pulling on themselves break things, but they should be zero

    dist_squareds[ignore] = 1

    # calculate pull strengths
    # pulls[i, j] = the magnitude of the
    # free-fall acceleration of planet i caused by planet j
    pulls = masses[p2s] * GRAV_CONSTANT / dist_squareds

    # get pull vectors and sum them for total accel
    # pull_vectors[i, j] is the vector of the free-fall
    # acceleration of planet i caused by planet j
    pull_vectors = pulls[..., None] * (
        distance_vectors / np.sqrt(dist_squareds)[..., None]
    )

    pull_vectors[ignore] = 0

    # sum up pulls from each other planet
    # accel_vectors[i] = the combined free-fall acceleration of planet j
    accel_vectors = np.sum(pull_vectors, axis=1)

    if lock is not None:
        accel_vectors -= accel_vectors[lock]

    return accel_vectors


def step(state, dt=1, masses=None, lock=None):
    accel_vectors = get_accel_vectors(state[:, 0], masses=masses, lock=lock)

    move(state, dt, accel_vectors)

    if lock is not None:
        state -= state[lock]


def simulate(state, time, masses=None, dt=1, lock=None):
    for i in range(0, time, dt):
        step(state[:, 0], dt=dt, masses=masses, lock=lock)


if __name__ == "__main__":
    state = np.zeros([3, 2, 2])
    state[:, 0] = np.array([[0, 0], [3, 0], [0, 4]])
    print(get_accel_vectors(state[:, 0]))
