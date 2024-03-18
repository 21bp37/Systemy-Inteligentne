import numpy as np
import typing

import pygame

from physics import Physics


class CelestialObject(Physics):
    def __init__(self, name: str, pos: typing.List, radius: typing.Union[int, float],
                 color: typing.Tuple[int, int, int],
                 mass: typing.Union[float, int], *, initial_velocity: typing.List[float] = None):
        self.name = name
        self.pos: np.ndarray = np.array(pos)
        self.radius: float = float(radius)
        self.color = color
        self.mass = float(mass)
        self.orbit_history: typing.List = []
        self.velocity: np.ndrray = np.array([0, 0]) if not initial_velocity else np.array(initial_velocity)

    def calculate_force(self, other: 'CelestialObject'):
        dist = np.linalg.norm(self.pos - other.pos)
        force = self.G * self.mass * other.mass / np.pow(dist, 2)
        angle = np.atan2(dist)
        fx = np.cos(angle) * force
        fy = np.sin(angle) * force
        return np.array(fx, fy)

    def calculate_total_force(self, objects: typing.List['CelestialObject']):
        positions = np.array([body.pos for body in objects if body is not self])
        masses = np.array([body.mass for body in objects if body is not self])
        delta_positions = positions - self.pos
        # print(self)
        r_squared = np.sum(delta_positions ** 2, axis=1)
        distances = np.linalg.norm(delta_positions, axis=1)
        force_magnitudes = self.G * self.mass * masses / r_squared
        force_vectors = force_magnitudes[:, np.newaxis] * delta_positions / distances[:, np.newaxis]
        total_force = np.sum(force_vectors, axis=0)
        return total_force[0], total_force[1]

    def __update_velocity(self, objects):
        total_fx, total_fy = self.calculate_total_force(objects)
        acceleration = np.array([total_fx / self.mass, total_fy / self.mass])
        self.velocity = np.add(acceleration * self.TIMESTEP, self.velocity)
        # print(self.velocity)

    def update_position(self, objects):
        objects_ex_self = [obj for obj in objects if obj is not self]
        self.__update_velocity(objects_ex_self)
        # self.orbit_history.append(np.copy(self.pos))
        self.pos = np.add(self.pos, self.velocity * self.TIMESTEP)

    def draw(self, window, move_x, move_y, draw_line, screen_size, scaling_factor=1.0):
        scale = self.SCALE * scaling_factor
        radius = self.radius * scaling_factor
        w, h = screen_size
        pos = np.add(self.pos, np.array([w / 2, h / 2]))
        self.orbit_history.append(pos)
        pos = pos * scale
        # print(self, self.pos)
        if len(self.orbit_history) > 2:
            updated_points = []
            for point in self.orbit_history:
                x, y = point * scale
                # x = x * self.SCALE + w / 2
                # y = y * self.SCALE + h / 2
                updated_points.append((x + move_x, y + move_y))
            if draw_line:
                pygame.draw.lines(window, self.color, False, updated_points, 1)
        pygame.draw.circle(window, self.color, (pos[0] + move_x, pos[1] + move_y), radius)

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


if __name__ == '__main__':
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600

    pygame.init()
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    earth_pos = [400 / Physics.SCALE + Physics.AU, 300 / Physics.SCALE]
    moon_pos = [earth_pos[0] + 0.002603 * Physics.AU, earth_pos[1]]
    objects = [
        CelestialObject(name="Moon", pos=moon_pos, radius=1,
                        color=(255, 255, 255), mass=7.34767309e22,
                        initial_velocity=[0, 29784 + 1022]),
        CelestialObject(name="Earth", pos=earth_pos, radius=3.7,
                        color=(0, 0, 255), mass=5.972e24,
                        initial_velocity=[0, 29784]),

        CelestialObject(name="Sun", pos=[400 / Physics.SCALE, 300 / Physics.SCALE], radius=30, color=(255, 255, 0),
                        mass=1.989e30),
    ]
    running = True
    scaling_factor = 1.0

    panning = False
    panning_offset = np.array([0, 0])
    panning_start = np.array([0, 0])
    while running:
        window.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    panning = True
                    panning_start = np.array(pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    panning = False
            elif event.type == pygame.MOUSEMOTION:
                if panning:
                    current_pos = np.array(pygame.mouse.get_pos())
                    panning_offset += current_pos - panning_start
                    panning_start = current_pos
            elif event.type == pygame.MOUSEWHEEL:
                scaling_factor = 1
                Physics.SCALE *= 1.25 if event.y > 0 else 0.75
                # Adjust panning offset to keep the current position under the mouse cursor
                # panning_offset = current_pos - scaling_factor * (current_pos - old_pan_offset)

        for obj in objects:
            obj.update_position(objects)
            obj.draw(window, panning_offset[0], panning_offset[1], True, (SCREEN_WIDTH, SCREEN_HEIGHT), scaling_factor)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
