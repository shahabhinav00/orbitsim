import numpy as np
import pickle
from sim_v2_partial import simulate_RK45_partial

with open("data/sol_solution_2023_0126_LX_1Y_64.p", "rb") as file:
	sol_solution = pickle.load(file)

sol_masses = np.load("data/sol_masses.npy")
selection = np.array([0, 1, 2, 6])
sol_masses = sol_masses[selection]

satellites = np.empty((1, 2, 3))
satellites[0] = np.array([
[-2.810901171854900E+07, 1.446571690009307E+08, 2.641749841572344E+04],
[-2.348481890142550E+01, -1.998451221644630E+00, 2.381631612165021E+00]
])

satellites[0] /= 1000

satellites[0] += np.load("data/sol_state_vectors.npy")[0]

sat_solution = simulate_RK45_partial(
	satellites, 
	365 * 24 * 3600, 
	sol_solution, 
	masses=sol_masses, 
)

with open("data/hubble_test.p", "wb") as file:
	pickle.dump(sat_solution, file)