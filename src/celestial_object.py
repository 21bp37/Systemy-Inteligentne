import numpy as np
import typing

import pygame

from src.physics import Physics


class CelestialObject(Physics):
    MAX_ORBIT_HISTORY_SIZE = 200

    def __init__(self, name: str, pos: typing.List, radius: typing.Union[int, float],
                 color: typing.Tuple[int, int, int],
                 mass: typing.Union[float, int], *, initial_velocity: typing.List[float] = None,
                 parent: typing.Optional['CelestialObject'] = None):
        self.softening_radius = 1e-5
        self.objects.append(self)
        self.name = name
        self.radius: float = float(radius) * 1000  # km
        self.color = color
        self.mass = float(mass)
        self.orbit_history: np.ndarray = np.zeros((0, 2))
        self.pos: np.ndarray = np.array(pos, dtype=np.float64)
        self.parent: typing.Optional['CelestialObject'] = parent
        self.velocity: np.array = np.array([0, 0], dtype=np.float64) if not initial_velocity else np.array(
            initial_velocity)
        if parent:
            parent_velocity = np.array(parent.velocity, dtype=float)
            self.velocity = np.array(initial_velocity, dtype=float) + parent_velocity
            self.pos += parent.pos

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
        total_force = np.sum(force_vectors, axis=0, dtype=np.float64)
        return total_force[0], total_force[1], distances

    def __update_velocity(self, total_fx, total_fy):
        acceleration = np.array([total_fx / self.mass, total_fy / self.mass])
        self.velocity = np.add(acceleration * self.TIMESTEP, self.velocity)

    def update_position(self, total_fx, total_fy):
        self.__update_velocity(total_fx, total_fy)
        self.pos = np.add(self.pos, self.velocity * self.TIMESTEP)
        # print(self.pos)

    def draw(self, window, move_x, move_y, draw_line, screen_size, scaling_factor=1.0):
        scale = self.SCALE * scaling_factor
        radius = max(self.radius, 1) * scaling_factor / self.SCALE
        w, h = screen_size
        pos = np.add(self.pos, np.array([w / 2, h / 2]))
        self.orbit_history = np.vstack((self.orbit_history[-(self.MAX_ORBIT_HISTORY_SIZE - 1):], pos))
        pos = pos * scale

        if len(self.orbit_history) > 2:
            updated_points = np.dot(self.orbit_history, np.array([[scale, 0], [0, scale]])) + np.array(
                [[move_x, move_y]])
            if draw_line:
                pygame.draw.lines(window, self.color, False, updated_points.astype(int), 1)
        # print(radius)
        pygame.draw.circle(window, self.color, (int(pos[0] + move_x), int(pos[1] + move_y)), radius)

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


if __name__ == '__main__':
    from src.draw import draw

    draw()
