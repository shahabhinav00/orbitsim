from sim_v2 import get_accel_vectors
from sim_v2_advanced import simulate_RK45, simulate_EUL
from sim_v2_partial import combine_solutions
import numpy as np
import pygame
import pickle

# displays things
# sizes is display size

rng = np.random.default_rng()


# randomly generates colors
def generate_colors(num):
    colors_raw = rng.integers(0, 2**24, num, dtype=np.uint32)
    colors = np.empty((num, 3), dtype=np.uint8)
    for i in range(3):
        colors[:, i] = np.bitwise_and(np.right_shift(colors_raw, i * 8), 255)

    return colors


def draw_arrow(surface, pos, vect, color, width=1):
    if np.sum(np.abs(vect)) == 0:
        return
    end = pos + vect
    direction = vect / np.sqrt(np.sum(vect**2))

    pygame.draw.line(surface, color, pos, end, width=width)

    offset = 10 * np.array([-direction[1], direction[0]])

    points = [end, end - (direction * 10) + offset, end - (direction * 10) - offset]

    pygame.draw.polygon(surface, color, points=points)


def draw(
    state,
    surface,
    sizes=None,
    masses=None,
    grav_scale=0,
    vel_scale=0,
    colors=None,
    arrow_scale=200,
    dist_scale=10,
    lock=None,
):
    num_planets = len(state)

    center = np.array([surface.get_width(), surface.get_height()]) / 2

    if state.shape[2] > 2:
        state = state[..., :2]

    if sizes is None:
        sizes = np.ones(num_planets)

    if colors is None:
        colors = generate_colors(num_planets)

    for i in range(num_planets):
        pygame.draw.circle(
            surface, colors[i], state[i, 0] * dist_scale + center, sizes[i]
        )

    if grav_scale > 0:
        accel_vectors = get_accel_vectors(state[:, 0], masses=masses, lock=lock)
        for i in range(num_planets):
            draw_arrow(
                surface,
                state[i][0] * dist_scale + center,
                accel_vectors[i] * grav_scale * masses[i],
                [255, 0, 0],
            )

    if vel_scale > 0:
        for i in range(num_planets):
            draw_arrow(
                surface,
                state[i][0] * dist_scale + center,
                state[i][1] * vel_scale,
                [0, 0, 255],
            )


# construct on the surface and iterate
# remember to clear the screen and flip the display
def draw_ode_solution(solution, surface, time, dt=1, lock=None, **kwargs):
    for t in range(0, time, dt):
        state = solution(t)
        if lock is not None:
            state -= state[lock]
        draw(state, surface, lock=lock, **kwargs)
        yield dt


if __name__ == "__main__":
    state = np.load("data/sol_state_vectors.npy")
    masses = np.load("data/sol_masses.npy")

    colors = np.array(
        [
            [47, 106, 105],
            [100, 100, 100],
            [255, 128, 0],
            [26, 26, 26],
            [230, 230, 230],
            [153, 61, 0],
            [176, 127, 53],
            [176, 143, 54],
            [85, 128, 170],
            [54, 104, 150],
            [255, 0, 0]
        ]
    )

    sizes = np.array([5, 3, 15, 2, 4, 4, 8, 7, 6, 5, 1])

    sizes *= 3

    selection = np.array([0, 1, 2, 6, -1])
    sizes = sizes[selection]
    colors = colors[selection]
    masses = masses[selection]

    pygame.init()

    lock = None

    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()

    running = True

    time = 10000000
    # dist_scale = 0.05 # LEO
    # dist_scale = 0.0007 # earth-moon
    dist_scale = 0.0000015 # inner planets
    # dist_scale = 0.0000001 # outer planets

    #solution = simulate_RK45(state, time, masses=masses, lock=lock)

    with open("data/sol_solution_2023_0126_L0_1Y_64.p", "rb") as file:
        sol_planets = pickle.load(file)

    with open("data/hubble_test.p", "rb") as file:
        sol_sats = pickle.load(file)


    solution = combine_solutions(sol_planets, sol_sats)

    print(solution(0))

    iterator = draw_ode_solution(
        solution,
        screen,
        time,
        dt=60,
        dist_scale=dist_scale,
        masses=np.append(masses, 1),
        sizes=sizes,
        colors=colors,
        vel_scale=1,
        grav_scale=10**-18,
        #lock=0,
    )

    for output in iterator:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if not running:
            break

        # delta-time is 1 hour
        # 60 FPS, so speed is 216000x real-time

        pygame.display.flip()

        screen.fill((255, 255, 255))

        clock.tick(60)

    pygame.quit()
