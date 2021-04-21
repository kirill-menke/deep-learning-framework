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
            x = np.arange(self.res)
            X, Y = np.meshgrid(x, x)

            mx, my = self.pos
            pts = (X - mx) ** 2 + (Y - my) ** 2 <= self.r ** 2
            self.output = np.where(pts, 1, 0)
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')


class Spectrum:

    def __init__(self, res: int):
        self.res = res
        self.output = None

    def draw(self):
        if not self.output:
            left = np.linspace((0, 0, 1), (0, 1, 1), self.res)
            right = np.linspace((1, 0, 0), (1, 1, 0), self.res)
            self.output = np.array([np.linspace(l, r, self.res) for l, r in zip(left, right)])
        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
