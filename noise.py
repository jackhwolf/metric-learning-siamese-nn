import numpy as np

def Noise(distance):
    noisy = 1
    distance = np.clip(distance, -30, 30)
    scaled_distance = 1 / (1 + np.exp(-1 * distance))
    if np.random.rand() > scaled_distance:
        noisy = -1
    return scaled_distance, noisy
