from sim_v2_advanced import simulate_RK45, simulate_EUL
import numpy as np
import pickle

state = np.load("data/sol_state_vectors.npy")
masses = np.load("data/sol_masses.npy")

# only earth, moon, sun, and jupiter
selection = np.array([0, 1, 2, 6])
state = state[selection]
masses = masses[selection]

time = 1 * 365 * 24 * 60 * 60

solution = simulate_RK45(state, time, masses=masses)

file = open("data/sol_solution_2023_0126_LX_1Y_64", "wb")
pickle.dump(solution, file)
file.close()
