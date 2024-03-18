from __future__ import annotations

import abc


class Physics:
    AU = 149.6e6 * 1000  # Astronomical unit
    G = 6.67428e-11  # Gravitational constant
    TIMESTEP = 1 * 24 * 1
    SCALE = 1250 / AU

    @abc.abstractmethod
    def calculate_force(self, other):
        raise NotImplementedError

    @classmethod
    def update_scale(cls, new_scale):
        cls.SCALE = cls.SCALE * new_scale
