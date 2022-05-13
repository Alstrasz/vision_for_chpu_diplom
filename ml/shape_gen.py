import this

import matplotlib.pyplot as plt
import numpy as np


class ShapeGen:
    def __init__(self):
        self.dots = dict()  # key -> array of 2d dots

    def add_dot(self, key, dot: [int, int]):
        if key in self.dots:
            self.dots[key].append(dot)
        else:
            self.dots[key] = [dot]

    def get_dots(self, key):
        if key in self.dots:
            return self.dots[key]
        else:
            return []

    def remove_dots(self, key):
        self.dots.pop(key, [])

    def get_len(self, key):
        if key in self.dots:
            return len(self.dots[key])
        else:
            return 0

    def get_image(self, key):
        formatted = np.array(self.dots[key]).T
        plt.plot(formatted[0], formatted[1], 'ro')
        plt.tight_layout()
        plt.savefig('./server/plot_' + str(key) + '_' + str(self.get_len(key) - 1) + '.png', vmax=4, vmin=4, hmax=3, hmin=3)

    def is_complete(self, key):
        if key not in self.dots:
            return False
        shape = self.dots[key]
        if len(shape) < 4:
            return False
        dist_0 = np.sqrt((shape[0][0] - shape[1][0]) ** 2 + (shape[0][1] - shape[1][1]) ** 2)
        dist_1 = np.sqrt((shape[1][0] - shape[2][0]) ** 2 + (shape[1][1] - shape[2][1]) ** 2)
        dist_3 = np.sqrt((shape[0][0] - shape[-1][0]) ** 2 + (shape[0][1] - shape[-1][1]) ** 2)
        if dist_3 < dist_0 + dist_1:
            return True
        return False
