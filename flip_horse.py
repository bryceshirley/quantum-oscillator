import math
from time import time

from toolbox_talk.data import get_initial_state
from toolbox_talk.operators import lie_trotter_step
from toolbox_talk.physics import get_propagators
from toolbox_talk.plotting import plot_state

N = 512
L = 10.0
T = math.pi
Nt = 100

psi_0 = get_initial_state(N, L, state_image="horse", blur=5.0)
V, K = get_propagators(N, L)


plot_state(psi_0, title="Initial Quantum State |psi|", L=L, close=True)
time_start = time()
for step in range(Nt):
    dt = T / Nt
    psi_0 = lie_trotter_step(psi_0, V, K, dt)
time_end = time()
print(f"Evolution completed in {time_end - time_start:.2f} seconds.")

plot_state(psi_0, title="Final Quantum State |psi| after Evolution", L=L)
