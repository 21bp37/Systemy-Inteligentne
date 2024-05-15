import pygame
import numpy as np
from src.physics import Physics
from src.celestial_object import CelestialObject


def draw():
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600

    pygame.init()
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
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
