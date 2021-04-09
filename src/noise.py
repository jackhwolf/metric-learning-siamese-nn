import numpy as np

# mu is P(label) == 1
def Noise(distance):
    true = np.sign(distance)
    noisy = 1
    mu = 1 / (1 + np.exp(-1*true))
    if np.random.rand() > mu:
        noisy = -1
    return mu, true, noisy
