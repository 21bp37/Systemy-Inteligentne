from __future__ import annotations

import abc

import numpy as np


class Physics:
    AU = 149.6e6 * 1000
    G = 6.67428e-11
    TIMESTEP = 2*1800 * 24 * 1
    SCALE = 125 / AU
    objects = []

    @abc.abstractmethod
    def calculate_force(self, other):
        raise NotImplementedError

    @classmethod
    def update_scale(cls, new_scale):
        cls.SCALE = cls.SCALE * new_scale


