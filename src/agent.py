import numpy as np
import pygame
# from physics import Physics
from src.celestial_object import CelestialObject
import typing


class Agent(CelestialObject):
    def __init__(self, name: str, pos: typing.List, radius: typing.Union[int, float],
                 color: typing.Tuple[int, int, int], mass: typing.Union[float, int],
                 initial_velocity: typing.List[float] = None,
                 parent: typing.Optional['CelestialObject'] = None, fuel_capacity: float = 1000,
                 thrust_force: float = 0.5, thrust_direction: typing.List[float] = None):

        super().__init__(name, pos, radius, color, mass, initial_velocity=initial_velocity, parent=parent)
        self.fuel_capacity = fuel_capacity
        self.fuel_level = fuel_capacity
        self.thrust_force = thrust_force
        self.thrust_direction = thrust_direction if thrust_direction is not None else [0, 0]

    def update_position(self, total_fx, total_fy):
        total_fx += self.thrust_direction[0] * self.thrust_force
        total_fy += self.thrust_direction[1] * self.thrust_force
        super().update_position(total_fx, total_fy)
        self.fuel_level -= np.linalg.norm(self.thrust_direction)

    def set_thrust_direction(self, direction: typing.List[float]):
        self.thrust_direction = direction

    def get_fuel_level(self) -> float:
        return self.fuel_level

    def draw(self, window, move_x, move_y, draw_line, screen_size, scaling_factor=1.0):
        scale = self.SCALE * scaling_factor
        radius = self.radius * scaling_factor
        w, h = screen_size
        pos = np.add(self.pos, np.array([w / 2, h / 2]))
        self.orbit_history.append(pos)
        pos = pos * scale

        pygame.draw.circle(window, self.color, (pos[0] + move_x, pos[1] + move_y), radius)
        rect_x = int(pos[0] - radius / 2)
        rect_y = int(pos[1])
        rect_width = int(radius)
        rect_height = int(radius / 2)

        # print((rect_x, rect_y, rect_width, rect_height))
        pygame.draw.rect(window, (100, 100, 100), (rect_x, rect_y, rect_width, rect_height))

        if self.thrust_direction[0] != 0 or self.thrust_direction[1] != 0:
            thruster_length = radius / 2
            thruster_width = radius / 4
            thruster_x = pos[0] + move_x + thruster_length * self.thrust_direction[0]
            thruster_y = pos[1] + move_y + thruster_length * self.thrust_direction[1]
            pygame.draw.line(window, (255, 0, 0), (int(pos[0] + move_x), int(pos[1] + move_y)),
                             (int(thruster_x), int(thruster_y)), int(thruster_width))
        if len(self.orbit_history) > 2:
            updated_points = []
            for point in self.orbit_history:
                x, y = point * scale
                # x = x * self.SCALE + w / 2
                # y = y * self.SCALE + h / 2
                updated_points.append((x + move_x, y + move_y))
            if draw_line:
                pygame.draw.lines(window, self.color, False, updated_points, 1)

    def draw_info(self, window, screen_size):
        font = pygame.font.SysFont('Arial', 20)
        fuel_text = font.render(f'Fuel: {self.fuel_level:.2f}', True, (255, 255, 255))
        window.blit(fuel_text, (screen_size[0] - 150, 20))

    def handle_events(self):
        keys = pygame.key.get_pressed()
        thrust_direction = [0, 0]
        if keys[pygame.K_w]:
            thrust_direction[1] -= 1
        if keys[pygame.K_s]:
            thrust_direction[1] += 1
        if keys[pygame.K_a]:
            thrust_direction[0] -= 1
        if keys[pygame.K_d]:
            thrust_direction[0] += 1

        self.set_thrust_direction(thrust_direction)
