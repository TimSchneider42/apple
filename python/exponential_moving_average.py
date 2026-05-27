import numpy as np


class ExponentialMovingAverage:
    def __init__(self, alpha: float | np.floating):
        self.alpha = alpha
        self.value = None

    def add(self, value: float | np.floating):
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * value
        return self.value

    def __call__(self, value: float | np.floating):
        return self.add(value)

    def clear(self):
        self.value = None
