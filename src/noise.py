import numpy as np

# mu is P(label) == 1
def Noise(distance):
    true = np.sign(distance)
    noisy = true
    mu = 1 / (1 + np.exp(distance))
    if np.random.rand() > mu:
        noisy *= -1
    return mu, true, noisy
