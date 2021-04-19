import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


class Checker:

    def __init__(self, res: int, tile_size: int):
        assert res % (2 * tile_size) == 0

        self.res = res
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        if not self.output:
            start = np.repeat(np.repeat(np.array([[0, 1], [1, 0]]), self.tile_size, axis=0),
                              self.tile_size, axis=1)
            reps = self.res // self.tile_size // 2
            self.output = np.tile(start, (reps, reps))
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')


class Circle:

    def __init__(self, res: int, r: int, pos: Tuple[int, int]):
        assert r <= res // 2
        assert all(0 <= p < res for p in pos)

        self.res = res
        self.r = r
        self.pos = pos
        self.output = None

    def draw(self):
        if not self.output:
            # https://stackoverflow.com/a/29330486/9505725
            x = np.arange(0, self.res)
            X, Y = np.meshgrid(x, x)

            mx, my = self.pos
            pts = (X - mx) ** 2 + (Y - my) ** 2 <= self.r ** 2

            self.output = np.zeros(X.shape)
            self.output[pts] = 1
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')


class Spectrum:

    def __init__(self, res: int):
        self.res = res
        self.output = None

    def draw(self):
        if not self.output:
            r = g = np.linspace(0, 1, self.res)
            b = np.linspace(1, 0, self.res)
            R, G = np.meshgrid(r, g)
            B, _ = np.meshgrid(b, g)

            R = np.expand_dims(R, 2)
            G = np.expand_dims(G, 2)
            B = np.expand_dims(B, 2)

            self.output = np.concatenate((R, G, B), axis=2)
        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
