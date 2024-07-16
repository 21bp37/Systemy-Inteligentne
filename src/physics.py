from __future__ import annotations

import abc

# import numpy as np


class Physics(abc.ABC):
    AU = 149.6e6 * 1000  # m
    G = 6.67428e-11
    # TIMESTEP = 1800 * 24 * 1
    TIMESTEP = 3600*0.5
    objects = []
    W = 800
    H = 800
    SCALE_X = W / AU  # 800px - 1AU
    SCALE_Y = H / AU  # 800px - 1AU
    SCREEN_CENTER = (W / 2, H / 2)

    @abc.abstractmethod
    def calculate_total_force(self, other):
        raise NotImplementedError
