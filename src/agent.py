import numpy as np
import pygame
# from physics import Physics
from src.celestial_object import CelestialObject
import typing


class Agent(CelestialObject):
    celestial_objects = np.copy(CelestialObject.objects)

    def __init__(self, name: str, pos: typing.List, radius: typing.Union[int, float],
                 color: typing.Tuple[int, int, int], mass: typing.Union[float, int],
                 initial_velocity: typing.List[float] = None,
                 parent: typing.Optional['CelestialObject'] = None, fuel_capacity: float = 1000,
                 thrust_force: float = 0.5, thrust_direction: typing.List[float] = None, orientation=0, target=None,
                 window=None):

        super().__init__(name, pos, radius, color, mass, initial_velocity=initial_velocity, parent=parent)
        self.fuel_capacity = fuel_capacity
        self.fuel_level = fuel_capacity
        self.thrust_force = thrust_force * 1
        self.thrust_direction = thrust_direction if thrust_direction is not None else [0, 0]
        self.orientation = orientation
        self.score = 0
        self.target = np.array([0, 0]) if target is None else np.array(target)
        self.celestial_objects = [obj for obj in CelestialObject.objects if not isinstance(obj, Agent)]
        self.window = window
        self.next_hist = []

    def distance_to_target(self):
        return np.linalg.norm(self.pos - self.target)

    @staticmethod
    def collision(distances) -> bool:
        collisions = distances <= 0
        return collisions.any().astype(bool)

    def distances(self):
        ps, rad = zip(*map(lambda x: (x.pos, x.radius), self.celestial_objects))
        # print('---')
        # print(self.pos)
        distances = np.linalg.norm(np.array(ps) - np.array(self.pos), axis=1) - np.array(rad)
        return np.maximum(distances, 0)  # [m]

    def update_position(self, total_fx, total_fy):
        velocity_angle = np.arctan2(total_fy, total_fx)
        self.orientation = velocity_angle
        total_fx += self.thrust_direction[0] * self.thrust_force
        total_fy += self.thrust_direction[1] * self.thrust_force
        super().update_position(total_fx, total_fy)
        self.fuel_level -= np.linalg.norm(self.thrust_direction)
        # self.calculate_score()

    def set_thrust_direction(self, direction: typing.List[float]):
        rotated_thrust_direction = [
            np.cos(self.orientation) * direction[0] - np.sin(self.orientation) * direction[1],
            np.sin(self.orientation) * direction[0] + np.cos(self.orientation) * direction[1]
        ]
        self.thrust_direction = rotated_thrust_direction

    def draw(self, window, move_x, move_y, draw_line, screen_size, scaling_factor=1.0):
        super().draw(window, move_x, move_y, draw_line, screen_size, scaling_factor)
        # radius = self.radius * scaling_factor
        scale = self.SCALE * scaling_factor
        w, h = screen_size
        pos = np.add(self.pos, np.array([w / 2, h / 2])) * scale

        start_point = (pos[0] + move_x, pos[1] + move_y)
        sc = 15
        end_point = (start_point[0] + self.thrust_direction[0] * sc,
                     start_point[1] + self.thrust_direction[1] * sc)
        pygame.draw.line(window, (255, 128, 200), start_point, end_point, 2)

        # target
        # print(target_pos)
        target_pos = np.add(self.target, np.array([w / 2, h / 2])) * scale
        # print(target_pos)
        print(target_pos)
        pygame.draw.circle(window, (163, 168, 225), (int(target_pos[0] + move_x),
                                                     int(target_pos[1] + move_y)), 3)
        # print((rect_x, rect_y, rect_width, rect_height))
        if len(self.next_hist) > 2:
            updated_points_next = np.dot(np.array(self.next_hist), np.array([[scale, 0], [0, scale]])) + np.array(
                [[move_x, move_y]])
            pygame.draw.lines(window, (128, 0, 128, 80), False, updated_points_next.astype(int), 1)
        self.draw_info(window, (w, h))

    def draw_info(self, window, screen_size):
        font = pygame.font.SysFont('Arial', 20)
        fuel_text = font.render(f'Fuel: {self.fuel_level:.2f}', True, (255, 255, 255))
        window.blit(fuel_text, (screen_size[0] - 150, 20))
        velocity_text = font.render(
            f"Velocity: {self.velocity[0] / 1000:.2f}, {self.velocity[1] / 1000:.2f} km/s", True,
            (255, 255, 255))
        window.blit(velocity_text, (10, 10))
        distance_text = font.render(
            f"Distance: {np.linalg.norm(self.pos - self.target) / self.AU} AU", True,
            (255, 255, 255))
        window.blit(distance_text, (10, 30))

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

    def simulate_next(self, n: int) -> np.ndarray:
        pos = np.copy(self.pos)
        pos_history = [pos]
        velocity = self.velocity
        for i in range(n):
            fx, fy = self.calculate_total_force_static(pos)
            # velocity_angle = np.arctan2(fx, fy)
            acceleration = np.array([fx / self.mass, fy / self.mass])
            velocity = np.add(acceleration * self.TIMESTEP, velocity)
            pos = np.add(pos, velocity * self.TIMESTEP)
            pos_history.append(pos)
        self.next_hist = pos_history
        # print(self.window)
        # print(pos_history)

        return np.array(pos_history)

    def calculate_total_force_static(self, pos):
        objects = self.celestial_objects
        positions, masses = zip(*map(lambda x: (x.pos, x.mass), objects))
        positions = np.array(positions, dtype=np.float64)
        masses = np.array(masses, dtype=np.float64)
        delta_positions = positions - pos
        r_squared = np.sum(delta_positions ** 2, axis=1)
        distances = np.sqrt(np.where(r_squared == 0, 1, r_squared))
        force_magnitudes = self.G * self.mass * masses / (r_squared + self.softening_radius ** 2)
        force_vectors = force_magnitudes[:, np.newaxis] * delta_positions / distances[:, np.newaxis]
        total_force = np.sum(force_vectors, axis=0, dtype=np.float64)
        return total_force[0], total_force[1]
