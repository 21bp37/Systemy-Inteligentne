import numpy as np
import typing

import pygame

from src.physics import Physics


class CelestialObject(Physics):
    def __init__(self, name: str, pos: typing.List, radius: typing.Union[int, float],
                 color: typing.Tuple[int, int, int],
                 mass: typing.Union[float, int], *, initial_velocity: typing.List[float] = None,
                 parent: typing.Optional['CelestialObject'] = None):
        self.objects.append(self)
        self.name = name
        self.radius: float = float(radius)
        self.color = color
        self.mass = float(mass)
        self.orbit_history: typing.List = []
        self.pos: np.ndarray = np.array(pos, dtype=np.float64)
        self.parent: typing.Optional['CelestialObject'] = parent
        self.velocity: np.ndrray = np.array([0, 0], dtype=np.float64) if not initial_velocity else np.array(
            initial_velocity)
        if parent:
            parent_velocity = np.array(parent.velocity, dtype=float)
            self.velocity = np.array(initial_velocity, dtype=float) + parent_velocity
            self.pos += parent.pos

    def calculate_force(self, other: 'CelestialObject'):
        dist_vec = other.pos - self.pos
        dist_sq = np.sum(dist_vec ** 2)
        softening_radius = 1e-5  # Softening parameter
        force = self.G * self.mass * other.mass / (dist_sq + softening_radius ** 2)
        angle = np.arctan2(dist_vec[1], dist_vec[0])  # Calculate angle using arctan2
        fx = np.cos(angle) * force
        fy = np.sin(angle) * force
        return np.array([fx, fy], dtype=np.float64)

    def calculate_total_force(self, objects: typing.List['CelestialObject']):
        positions = np.array([body.pos for body in objects if body is not self], dtype=np.float64)
        masses = np.array([body.mass for body in objects if body is not self], dtype=np.float64)
        delta_positions = positions - self.pos
        r_squared = np.sum(delta_positions ** 2, axis=1)
        distances = np.sqrt(r_squared)
        softening_radius = 1e-5  # Softening parameter
        force_magnitudes = self.G * self.mass * masses / (r_squared + softening_radius ** 2)
        force_vectors = force_magnitudes[:, np.newaxis] * delta_positions / distances[:, np.newaxis]
        total_force = np.sum(force_vectors, axis=0, dtype=np.float64)
        return total_force[0], total_force[1]

    def __update_velocity(self, total_fx, total_fy):
        acceleration = np.array([total_fx / self.mass, total_fy / self.mass])
        self.velocity = np.add(acceleration * self.TIMESTEP, self.velocity)

    def update_position(self, total_fx, total_fy):
        self.__update_velocity(total_fx, total_fy)

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
    sun_pos = [400 / Physics.SCALE, 300 / Physics.SCALE]
    sun = CelestialObject(name="Sun", pos=sun_pos, radius=30, color=(255, 255, 0), mass=1.989e30)

    earth = CelestialObject(name="Earth", pos=[1 * Physics.AU, 0], radius=3.7,
                            color=(0, 0, 255), mass=5.972e24,
                            initial_velocity=[0, 29784], parent=sun)

    moon = CelestialObject(name="Moon", pos=[0.002603 * Physics.AU, 0], radius=1,
                           color=(255, 255, 255), mass=7.34767309e22,
                           initial_velocity=[0, 1022], parent=earth)

    mercury_pos = [0.39 * Physics.AU, 0]
    venus_pos = [0.72 * Physics.AU, 0]
    mars_pos = [1.52 * Physics.AU, 0]
    jupiter_pos = [5.20 * Physics.AU, 0]
    saturn_pos = [9.58 * Physics.AU, 0]
    uranus_pos = [19.22 * Physics.AU, 0]
    neptune_pos = [30.05 * Physics.AU, 0]
    pluto_pos = [39.48 * Physics.AU, 0]

    # Define solar system objects for other planets
    mercury = CelestialObject(name="Mercury", pos=mercury_pos, radius=1.6,
                              color=(128, 128, 128), mass=3.3011e23,
                              initial_velocity=[0, 47870], parent=sun)
    venus = CelestialObject(name="Venus", pos=venus_pos, radius=3.8,
                            color=(255, 215, 0), mass=4.8675e24,
                            initial_velocity=[0, 35020], parent=sun)
    mars = CelestialObject(name="Mars", pos=mars_pos, radius=1.9,
                           color=(255, 0, 0), mass=6.417e23,
                           initial_velocity=[0, 24130], parent=sun)

    # Moons of Jupiter
    jupiter_initial_velocity = [0, 13070]  # Jupiter's initial velocity in m/s
    io_initial_velocity = [0, 17000]  # Io's initial velocity in m/s
    io_pos = [0.00282089577 * Physics.AU, 0]

    # Define celestial objects using the corrected initial velocities
    jupiter = CelestialObject(name="Jupiter", pos=jupiter_pos, radius=71.4,
                              color=(255, 165, 0), mass=1.898e27,
                              initial_velocity=jupiter_initial_velocity, parent=sun)

    io = CelestialObject(name="Io", pos=io_pos, radius=1,
                         color=(200, 200, 200), mass=8.931938e22,
                         initial_velocity=io_initial_velocity, parent=jupiter)
    saturn = CelestialObject(name="Saturn", pos=saturn_pos, radius=60.3,
                             color=(255, 215, 0), mass=5.683e26,
                             initial_velocity=[0, 9690], parent=sun)
    uranus = CelestialObject(name="Uranus", pos=uranus_pos, radius=25.6,
                             color=(173, 216, 230), mass=8.681e25,
                             initial_velocity=[0, 6810], parent=sun)
    neptune = CelestialObject(name="Neptune", pos=neptune_pos, radius=24.8,
                              color=(30, 144, 255), mass=1.024e26,
                              initial_velocity=[0, 5430], parent=sun)
    pluto = CelestialObject(name="Pluto", pos=pluto_pos, radius=1,
                            color=(192, 192, 192), mass=1.303e22,
                            initial_velocity=[0, 4749], parent=sun)

    # europa = CelestialObject(name="Europa", pos=[europa_distance_from_jupiter * Physics.AU, 0], radius=1,
    #                          color=(200, 200, 200), mass=4.799844e22,
    #                          initial_velocity=[0, 13740], parent=jupiter)

    # titan_distance_from_saturn = 1200e3 / Physics.AU  # Distance of Titan from Saturn in AU
    # rhea_distance_from_saturn = 1345e3 / Physics.AU  # Distance of Rhea from Saturn in AU
    #
    # titan = CelestialObject(name="Titan", pos=[titan_distance_from_saturn * Physics.AU, 0], radius=1,
    #                         color=(200, 200, 200), mass=1.3452e23,
    #                         initial_velocity=[0, 5570], parent=saturn)
    # rhea = CelestialObject(name="Rhea", pos=[rhea_distance_from_saturn * Physics.AU, 0], radius=1,
    #                        color=(200, 200, 200), mass=2.306518e21,
    #                        initial_velocity=[0, 5260], parent=saturn)

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

        forces = {}
        for obj in Physics.objects:
            force = obj.calculate_total_force(Physics.objects)
            forces[obj] = force

        for obj, force in forces.items():
            obj.update_position(*force)
            obj.draw(window, panning_offset[0], panning_offset[1], True, (SCREEN_WIDTH, SCREEN_HEIGHT), scaling_factor)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
