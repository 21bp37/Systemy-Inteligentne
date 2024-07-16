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
                 thrust_force: float = 0.2, thrust_direction: typing.List[float] = None, target=None,
                 window=None):

        super().__init__(name, pos, radius, color, mass, initial_velocity=initial_velocity, parent=parent)
        self.fuel_capacity = fuel_capacity
        self.fuel_level = fuel_capacity
        self.thrust_force = thrust_force * 1
        self.thrust_direction = thrust_direction if thrust_direction is not None else [0, 0]
        self.orientation = self.calc_orientation()
        self.score = 0
        self.target = np.array([0, 0]) if target is None else np.array(target)
        self.celestial_objects = [obj for obj in CelestialObject.objects if not isinstance(obj, Agent)]
        self.window = window
        self.next_hist = []
        self.target_points = [self.target]
        self.radius = 4
        self.reached = False
        # for obj in self.objects:
        #     if self.satellite is obj:
        #         continue
        #     obj: 'CelestialObject'
        #     lidars.append(self.satellite.create_lidar_vector(obj.pos, obj.radius))
        self.lidars = [self.calc_l_vector(obj.pos, obj.radius) for obj in self.objects if obj is not self]
        self.text_list = []
        # self.lidars = []

    def distance_to_target(self):
        return np.linalg.norm(self.pos - self.target)

    @staticmethod
    def collision(distances) -> bool:
        collisions = distances <= 0
        return collisions.any().astype(bool)

    def distances(self):
        ps, rad = zip(*map(lambda x: (x.pos, x.radius), self.celestial_objects))
        distances = np.linalg.norm(np.array(ps) - np.array(self.pos), axis=1) - np.array(rad)
        return np.maximum(distances, 0)  # [m]

    def calc_orientation(self):
        vel = self.velocity / np.linalg.norm(self.velocity)
        angle = np.arctan2(vel[1], vel[0]) % (2 * np.pi)
        return angle

    def update_position(self, total_fx, total_fy):
        # vel = self.velocity/np.linalg.norm(self.velocity)
        # dt = self.TIMESTEP
        total_fx += self.thrust_direction[0] * self.thrust_force
        total_fy += self.thrust_direction[1] * self.thrust_force
        velocity_angle = self.calc_orientation()
        self.orientation = velocity_angle
        super().update_position(total_fx, total_fy)
        velocity_angle = self.calc_orientation()
        self.orientation = velocity_angle
        ''
        self.fuel_level -= np.linalg.norm(self.thrust_direction)
        self.lidars = [self.calc_l_vector(obj.pos, obj.radius) for obj in self.objects if obj is not self]

    def set_thrust_direction(self, direction: typing.List[float]):
        # orientation = self.orientation
        orientation = 0
        rotated_thrust_direction = [
            np.cos(orientation) * direction[0] - np.sin(orientation) * direction[1],
            np.sin(orientation) * direction[0] + np.cos(orientation) * direction[1]
        ]
        self.thrust_direction = rotated_thrust_direction

    def draw(self, window):
        super().draw(window)
        scale_x = self.SCALE_X
        scale_y = self.SCALE_Y
        center = self.SCREEN_CENTER
        w, h = window.get_width(), window.get_height()
        # radius = 10
        pos = np.add(self.pos, np.array(center))
        self.orbit_history = np.vstack((self.orbit_history[-(self.MAX_ORBIT_HISTORY_SIZE - 1):], pos))
        pos[0] = pos[0] * scale_x
        pos[1] = pos[1] * scale_y
        pos = np.add(self.pos, np.array([w / 2, h / 2]))
        pos[0] *= scale_x
        pos[1] *= scale_y
        start_point = (pos[0], pos[1])
        sc = 15
        end_point = (start_point[0] + self.thrust_direction[0] * sc,
                     start_point[1] + self.thrust_direction[1] * sc)
        pygame.draw.line(window, (255, 128, 200), start_point, end_point, 2)
        vel = self.velocity / np.linalg.norm(self.velocity)
        end_point = (start_point[0] + vel[0] * sc,
                     start_point[1] + vel[1] * sc)
        pygame.draw.line(window, (255, 255, 200), start_point, end_point, 2)
        # wektory sil
        # try:
        #     force_magnitudes = [np.linalg.norm(vector) for vector in self.current_force_vectors]
        #     sorted_indices = np.argsort(force_magnitudes)[::-1]
        #     n_vecs = 4
        #     strongest_forces = [self.current_force_vectors[i] for i in sorted_indices[:n_vecs]]
        #     for i, force in enumerate(strongest_forces):
        #         force = force / np.linalg.norm(force)
        #         sc = (i + 2) * 15
        #         end_point = (start_point[0] + force[0] * sc, start_point[1] + force[1] * sc)
        #         pygame.draw.line(window, (255, 255, 255), start_point, end_point, 2)
        # except ValueError:
        #     pass

        sc = self.W
        line_surface = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        line_surface = line_surface.convert_alpha()
        draw_lidars = [self.calc_l_vector(obj.pos, obj.radius) for obj in self.objects if obj is not self]
        for vector in draw_lidars:
            end_point = (start_point[0] + vector[0] * sc, start_point[1] + vector[1] * sc)
            pygame.draw.line(line_surface, (255, 0, 0, 40), start_point, end_point, 2)
        vector = self.calc_l_vector(self.target)
        end_point = (start_point[0] + vector[0] * sc, start_point[1] + vector[1] * sc)
        pygame.draw.line(line_surface, (255, 180, 190, 40), start_point, end_point, 2)
        window.blit(line_surface, (0, 0))
        if len(self.next_hist) > 2:
            updated_points_next = np.dot(np.array(self.next_hist), np.array([[scale_x, 0], [0, scale_y]]))
            pygame.draw.lines(window, (128, 0, 128, 80), False, updated_points_next.astype(int), 1)
        self.draw_info(window, (w, h), start_point=start_point)

    def draw_info(self, window, screen_size, start_point=None):
        font = pygame.font.SysFont('Arial', 20)
        # fuel_text = font.render(f'Fuel: {self.fuel_level:.2f}', True, (255, 255, 255))
        # window.blit(fuel_text, (screen_size[0] - 150, 20))
        velocity_text = font.render(
            f"Prędkość: {self.velocity[0] / 1000:.2f}, {self.velocity[1] / 1000:.2f} km/s", True,
            (255, 255, 255))
        window.blit(velocity_text, (10, 10))
        # distance = np.linalg.norm(self.pos - self.target) / self.AU
        distance_text = font.render(
            f"Odległość: {np.linalg.norm(self.pos - self.target) / self.AU:.2f} AU", True,
            (255, 255, 255))
        window.blit(distance_text, (10, 30))

        score_text = font.render(
            f"Wynik: {self.score}", True,
            (255, 190, 190))
        window.blit(score_text, (10, 55))
        target_color = (255, 255, 255) if not self.reached else (30, 240, 30)
        target_pos = np.add(self.target, np.array([self.W / 2, self.H / 2]))
        target_pos[0] *= self.SCALE_X
        target_pos[1] *= self.SCALE_Y
        x_center = int(target_pos[0])
        y_center = int(target_pos[1])
        offset = 9
        # color = (255, 255, 255)
        pygame.draw.line(window, target_color, (x_center - offset, y_center - offset),
                         (x_center + offset, y_center + offset), 3)
        pygame.draw.line(window, target_color, (x_center - offset, y_center + offset),
                         (x_center + offset, y_center - offset), 3)
        font = pygame.font.SysFont('Arial', 14)
        text = font.render(f'[{self.target[0] / self.AU:.2f},{self.target[1] / self.AU:.2f}]', True, target_color)

        window.blit(text, (x_center + 1.5 * offset, y_center - 2.4 * offset))
        font = pygame.font.SysFont('Arial', 15)
        text = font.render(f'[{self.pos[0] / self.AU:.2f},{self.pos[1] / self.AU:.2f}]', True, (255, 255, 255))
        window.blit(text, (start_point[0] + 1.5 * offset, start_point[1] - 2.4 * offset))
        self.draw_text(window)

    def draw_text(self, window):
        font = pygame.font.SysFont('Arial', 18)
        for i, text in enumerate(self.text_list):
            txt = font.render(text, True, (255, 255, 255))
            # window.blit(fuel_text, (screen_size[0] - 150, 20))
            window.blit(txt, (self.W - 150, 55 + i * 20))

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

    # def simulate_next(self, n: int, td=False) -> np.ndarray:
    #     pos = np.copy(self.pos)
    #     pos_history = [pos]
    #     velocity = self.velocity
    #     thrust_direction = self.thrust_direction
    #     thrust_force = self.thrust_force
    #     for i in range(n):
    #         fx, fy = self.calculate_total_force_static(pos)
    #         if td:
    #             fx += thrust_direction[0] * thrust_force
    #             fy += thrust_direction[1] * thrust_force
    #         acceleration = np.array([fx / self.mass, fy / self.mass])
    #         velocity = np.add(acceleration * self.TIMESTEP, velocity)
    #         pos = np.add(pos, velocity * self.TIMESTEP)
    #         pos_history.append(pos)
    #     if not td:
    #         self.next_hist = pos_history
    #     return np.array(pos_history)
    #
    # def simulate_next_v(self, n: int, td=False) -> typing.Tuple:
    #     pos = np.copy(self.pos)
    #     pos_history = [pos]
    #     velocity = self.velocity
    #     thrust_direction = self.thrust_direction
    #     thrust_force = self.thrust_force
    #     for i in range(n):
    #         fx, fy = self.calculate_total_force_static(pos)
    #         if td:
    #             fx += thrust_direction[0] * thrust_force
    #             fy += thrust_direction[1] * thrust_force
    #         acceleration = np.array([fx / self.mass, fy / self.mass])
    #         velocity = np.add(acceleration * self.TIMESTEP, velocity)
    #         pos = np.add(pos, velocity * self.TIMESTEP)
    #         pos_history.append(pos)
    #     return pos_history[-1], velocity

    # def calculate_total_force_static(self, pos):
    #     objects = self.celestial_objects
    #     positions, masses = zip(*map(lambda x: (x.pos, x.mass), objects))
    #     positions = np.array(positions, dtype=np.float64)
    #     masses = np.array(masses, dtype=np.float64)
    #     delta_positions = positions - pos
    #     r_squared = np.sum(delta_positions ** 2, axis=1)
    #     distances = np.sqrt(np.where(r_squared == 0, 1, r_squared))
    #     force_magnitudes = self.G * self.mass * masses / (r_squared + self.softening_radius ** 2)
    #     force_vectors = force_magnitudes[:, np.newaxis] * delta_positions / distances[:, np.newaxis]
    #     total_force = np.sum(force_vectors, axis=0, dtype=np.float64)
    #     return total_force[0], total_force[1]

    # def calc_l_vector(self, obj_pos: np.array, obj_radius: float = 0.0) -> np.array:
    #     ship_center_obj_vec = obj_pos - self.pos
    #     ship_obj_angle = np.arctan2(ship_center_obj_vec[..., 1], ship_center_obj_vec[..., 0])
    #     ship_obj_angle %= 2 * np.pi
    #     scale = (np.linalg.norm(ship_center_obj_vec) - obj_radius) * 2 / self.AU
    #     angle_unit = np.stack([np.cos(ship_obj_angle), np.sin(ship_obj_angle)], axis=-1) * scale
    #     # angle_unit = angle_unit / np.linalg.norm(angle_unit) # ????
    #     # print(angle_unit)
    #     return angle_unit
    #
    # def calc_l_vector_draw(self, obj_pos: np.array, obj_radius: float = 0.0) -> np.array:
    #     ship_center_obj_vec = obj_pos - self.pos
    #     ship_obj_angle = np.arctan2(ship_center_obj_vec[..., 1], ship_center_obj_vec[..., 0])
    #     ship_obj_angle %= 2 * np.pi
    #     scale = (np.linalg.norm(ship_center_obj_vec) - obj_radius) * 2 / self.AU
    #     angle_unit = np.stack([np.cos(ship_obj_angle), np.sin(ship_obj_angle)], axis=-1) * scale
    #     return angle_unit
    def calc_l_vector(self, pos: np.ndarray, r: float = 0.0) -> np.array:
        facing_vec = pos - self.pos
        angle = np.arctan2(facing_vec[..., 1], facing_vec[..., 0]) % (2 * np.pi)
        scale = (np.linalg.norm(facing_vec) - r) / self.AU
        angle_hat = np.stack([np.cos(angle), np.sin(angle)], axis=-1) * scale
        # angle_hat = angle_hat / np.linalg.norm(angle_hat) # ????
        # print(angle_unit)
        return angle_hat

    def angle_vector(self):
        return np.stack([np.cos(self.orientation), np.sin(self.orientation)], axis=-1)
