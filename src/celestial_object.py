import numpy as np
import typing

import pygame

from src.physics import Physics


class CelestialObject(Physics):
    MAX_ORBIT_HISTORY_SIZE = 64

    def __init__(self, name: str, pos: typing.List, radius: typing.Union[int, float],
                 color: typing.Tuple[int, int, int],
                 mass: typing.Union[float, int], *, initial_velocity: typing.List[float] = None,
                 parent: typing.Optional['CelestialObject'] = None, include_in_calcs=True):
        self.softening_radius = 1e-5
        self.perform_calc = include_in_calcs
        self.objects.append(self)
        self.name = name
        self.radius: float = radius
        self.radius_au = self.radius / Physics.AU
        self.color = color
        self.mass = float(mass)  # kg
        self.orbit_history: np.ndarray = np.zeros((0, 2))
        self.pos: np.ndarray = np.array(pos, dtype=np.float64)  # x,y w km.
        # 800px = 1au
        self.parent: typing.Optional['CelestialObject'] = parent
        self.velocity: np.array = np.array([0, 0], dtype=np.float64) if not initial_velocity else np.array(
            initial_velocity, dtype=np.float64)
        if parent:
            self.velocity += np.array(parent.velocity, dtype=np.float64)
            self.pos += parent.pos
        # 1px = 1/800AU
        self.draw_radius = self.radius_au * self.W
        self.current_force_vectors = None
        # print(self, self.draw_radius)
        # self.radius_km = 300_000

    @property
    def normalized_pos(self):
        pos_au = self.pos / Physics.AU
        return np.array([pos_au[0], pos_au[1]])

    def calculate_total_force(self, objects: typing.List['CelestialObject']):
        # positions = np.array([body.pos for body in objects if body is not self], dtype=np.float64)
        # masses = np.array([body.mass for body in objects if body is not self], dtype=np.float64)
        positions, masses = zip(*map(lambda x: (x.pos, x.mass), objects))
        positions = np.array(positions, dtype=np.float64)
        masses = np.array(masses, dtype=np.float64)
        delta_positions = positions - self.pos
        r_squared = np.sum(delta_positions ** 2, axis=1)
        distances = np.sqrt(np.where(r_squared == 0, 1, r_squared))
        force_magnitudes = self.G * self.mass * masses / (r_squared + self.softening_radius ** 2)
        force_vectors = force_magnitudes[:, np.newaxis] * delta_positions / distances[:, np.newaxis]
        self.current_force_vectors = force_vectors
        total_force = np.sum(force_vectors, axis=0, dtype=np.float64)
        return total_force[0], total_force[1], distances

    def __update_velocity(self, total_fx, total_fy):
        acceleration = np.array([total_fx / self.mass, total_fy / self.mass])
        self.velocity = np.add(acceleration * self.TIMESTEP, self.velocity)

    def update_position(self, total_fx, total_fy):
        if self.perform_calc:
            self.__update_velocity(total_fx, total_fy)
            self.pos = np.add(self.pos, self.velocity * self.TIMESTEP)
        # print(self.pos)

    def draw(self, window: typing.Union[pygame.Surface, pygame.SurfaceType]):
        scale_x = self.SCALE_X
        scale_y = self.SCALE_Y
        center = self.SCREEN_CENTER
        pos = np.add(self.pos, np.array(center))
        self.orbit_history = np.vstack((self.orbit_history[-(self.MAX_ORBIT_HISTORY_SIZE - 1):], pos))
        pos[0] = pos[0] * scale_x
        pos[1] = pos[1] * scale_y

        if len(self.orbit_history) > 2:
            updated_points = np.dot(self.orbit_history, np.array([[scale_x, 0], [0, scale_y]]))
            pygame.draw.lines(window, self.color, False, updated_points.astype(int), 1)
        # print(radius)
        try:
            pygame.draw.circle(window, self.color, (int(pos[0]), int(pos[1])), self.draw_radius)
        except TypeError:
            return

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


if __name__ == '__main__':
    from src.draw import draw, draw_sat
    from src.objects import initialize_objects
    from src.agent import Agent

    initialize_objects()
    earth = next((obj for obj in Physics.objects if obj.name.lower() == 'earth'), None)
    # print(earth)
    satellite = Agent(name="Satellite", pos=[0.05 * Physics.AU, 0], radius=1, color=(255, 128, 255), mass=1,
                      initial_velocity=[0, 822], parent=earth, fuel_capacity=3000,
                      target=[0.7 * Physics.AU, 0.2 * Physics.AU])
    satellite.draw_radius = 4
    for obj in Physics.objects:
        if obj is satellite:
            continue
        satellite.calc_l_vector(obj.pos, obj.radius)
    draw_sat(satellite)
