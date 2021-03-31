import numpy as np

# mu is P(label) == 1
def Noise(distance):
    noisy = 1
    scaled_distance = 1 / (1 + np.exp(-1 * distance))
    if np.random.rand() > scaled_distance:
        noisy = -1
    return scaled_distance, noisy
