import pygame

from src.physics import Physics
from src.celestial_object import CelestialObject


def draw():
    SCREEN_WIDTH = Physics.W
    SCREEN_HEIGHT = Physics.H

    pygame.init()
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    # clock = pygame.time.Clock()

    running = True

    while running:
        window.fill((0, 0, 0))
        forces = {}
        pygame.event.pump()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        for obj in CelestialObject.objects:
            force = obj.calculate_total_force(CelestialObject.objects)
            forces[obj] = (force[0], force[1])

        for obj, force in forces.items():
            # print(force)
            # print(force)
            obj: 'CelestialObject'
            obj.update_position(*force)
            obj.draw(window)

        pygame.display.flip()

    pygame.quit()


def draw_sat(satellite):
    SCREEN_WIDTH = Physics.W
    SCREEN_HEIGHT = Physics.H

    pygame.init()
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    clock = pygame.time.Clock()

    running = True

    while running:
        window.fill((0, 0, 0))
        forces = {}
        pygame.event.pump()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        for obj in CelestialObject.objects:
            force = obj.calculate_total_force(CelestialObject.objects)
            forces[obj] = (force[0], force[1])

        for obj, force in forces.items():
            # print(force)
            # print(force)
            obj: 'CelestialObject'
            obj.update_position(*force)
            obj.draw(window)

        # satellite.draw(window)
        pygame.display.flip()
        satellite.handle_events()

    pygame.quit()


if __name__ == '__main__':
    # sterowanie satelita wasd
    # from new.src.draw import draw, draw_sat
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
