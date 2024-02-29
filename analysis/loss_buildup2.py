from data_generation.advanced import simulate_RK45
from data_generation.partial import simulate_RK45_partial
from data_generation.orbit_generator import generate_state_vectors
from load_utils import load_model, accuracy
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import pygame
import sys

generate_new_data = False

if generate_new_data:
    initial_state = generate_state_vectors(100, dims=3, flat=True)
    np.save("data/AbhinavUpdateTemp.npy", initial_state)
else:
    initial_state = np.load("data/AbhinavUpdateTemp.npy")

planet_selection = np.array([0, 1, 2])
masses = np.load("data/sol_masses.npy")[planet_selection]
planet_state = np.load("data/sol_state_vectors.npy")[planet_selection]

future = 60 * 60 * 12


update_time = -2

if update_time == -1:
    update_time = int(input("Prediction target time: "))

elif update_time == -2:
    update_time = int(sys.argv[1])


cycles = future//update_time
planet_solution = simulate_RK45(
    planet_state,
    future,
    masses=masses,
    dt=1,
    lock=0
)

sat_solution = simulate_RK45_partial(
    initial_state,
    future,
    planet_solution,
    dt=1,
    masses=masses
)


model = load_model(f"saved_models/ANN_06_{update_time}_FINAL")

state = sat_solution(0)
coordinates = []

errors = []

for i in range(cycles + 1):
    time = i * update_time
    state = model.predict(state, verbose=0)

    print(state)

    x = int((state[0, 0, 0])/25) + 640  # scaling
    y = int((state[0, 0, 1])/25) + 360
    coordinates.append((x, y))

    errors.append(accuracy(sat_solution(time)[..., :2], state[..., :2]))

print(coordinates)
errors = np.array(errors)


# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True


def place_image(screen, pos, filename, size):
    img = pygame.transform.scale(pygame.image.load(filename), np.array(size) * 2)
    screen.blit(img, pos -  np.array(size))


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("white")

    # RENDER YOUR GAME HERE
    pygame.draw.lines(screen, [0, 0, 0], False, coordinates)
    place_image(screen, (640, 360), "data/earth.png", (40, 40))
    place_image(screen, coordinates[0], "data/sat_icon.png", (25,25))
    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()
