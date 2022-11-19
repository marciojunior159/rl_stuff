import numpy as np

class kArmedBandits:
    def __init__(self, k):
        self.arms = [ArmedBandit() for _ in range(k)]
        self.optimal = np.argmax([a.arm_mean for a in self.arms])

    def get_reward(self, i):
        return self.arms[i].reward()

class ArmedBandit:
    def __init__(self):
        self.arm_mean = np.random.normal(0, 1)
        
    def reward(self):
        return np.random.normal(self.arm_mean, 1)
