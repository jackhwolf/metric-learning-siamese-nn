import numpy as np

# mu is P(label) == 1
def Noise(distance):
    true = np.sign(-1 * distance)
    noisy = true
    mu = 1 / (1 + np.exp(-1 * distance))
    if np.random.rand() > mu:
        noisy *= -1
    return mu, true, noisy
