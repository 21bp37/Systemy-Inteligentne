import pygame
import numpy as np
from src.agent import Agent
from src.celestial_object import CelestialObject
from src import Physics


def main():
    pygame.init()

    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Satellite Simulation")

    clock = pygame.time.Clock()

    running = True
    scaling_factor = 1.0
    panning = False
    panning_offset = np.array([0, 0])
    panning_start = np.array([0, 0])
    sun_pos = [400 / Physics.SCALE, 300 / Physics.SCALE]
    sun = CelestialObject(name="Sun", pos=sun_pos, radius=30, color=(255, 255, 0), mass=1.989e30)

    earth = CelestialObject(name="Earth", pos=[1 * Physics.AU, 0], radius=3.7,
                            color=(0, 0, 255), mass=5.972e24,
                            initial_velocity=[0, 29784], parent=sun)

    moon = CelestialObject(name="Moon", pos=[0.002603 * Physics.AU, 0], radius=1,
                           color=(255, 255, 255), mass=7.34767309e22,
                           initial_velocity=[0, 1022], parent=earth)

    satellite = Agent(name="Satellite", pos=[0.002 * Physics.AU, 0], radius=1, color=(255, 0, 0), mass=1000,
                      initial_velocity=[0, 1022], parent=earth, fuel_capacity=10000)

    font = pygame.font.SysFont('Arial', 20)
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
                scaling_factor *= 1.25 if event.y > 0 else 0.75

            satellite.handle_events()

        forces = {}
        for obj in CelestialObject.objects:
            force = obj.calculate_total_force(CelestialObject.objects)
            forces[obj] = force

        for obj, force in forces.items():
            obj.update_position(*force)
            obj.draw(window, panning_offset[0], panning_offset[1], True, (SCREEN_WIDTH, SCREEN_HEIGHT), scaling_factor)

        satellite.draw_info(window, (SCREEN_WIDTH, SCREEN_HEIGHT))
        satellite.draw(window, panning_offset[0], panning_offset[1], True, (SCREEN_WIDTH, SCREEN_HEIGHT),
                       scaling_factor)

        velocity_text = font.render(
            f"Velocity: {satellite.velocity[0] / 1000:.2f}, {satellite.velocity[1] / 1000:.2f} km/s", True,
            (255, 255, 255))
        window.blit(velocity_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)


if __name__ == '__main__':
    main()
