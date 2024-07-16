import numpy as np

from src.physics import Physics
from src.celestial_object import CelestialObject


def config1():
    sun_pos = [0.5 * Physics.AU, 0.5 * Physics.AU]
    sun = CelestialObject(name="Sun", pos=sun_pos, radius=0.04 * Physics.AU, color=(255, 255, 0), mass=1.989e30,
                          include_in_calcs=False)
    # sun.draw_radius = 40
    # print(sun.pos)
    # print(sun.normalized_pos)

    # moon = CelestialObject(name="Moon", pos=[0.002603 * Physics.AU, 0], radius=1737,
    #                        color=(255, 255, 255), mass=7.34767309e22,
    #                        initial_velocity=[0, 1022], parent=earth)

    mercury_pos = [0.3 * Physics.AU, 0]
    r = np.linalg.norm(mercury_pos)
    mercury_vel = [0, 49870]
    phi = 4 / 3 * np.pi
    x_new = r * np.cos(phi)
    y_new = r * np.sin(phi)
    mercury_pos = [x_new, y_new]

    omega = np.linalg.norm(mercury_vel) / r
    vx_new = -omega * r * np.sin(phi)
    vy_new = omega * r * np.cos(phi)
    mercury_vel = [vx_new, vy_new]

    venus_pos = [-0.4 * Physics.AU, 0]
    p1 = CelestialObject(name="p1", pos=mercury_pos, radius=0.014 * Physics.AU,
                         color=(128, 128, 128), mass=3e23,
                         initial_velocity=mercury_vel, parent=sun)
    p2 = CelestialObject(name="p2", pos=venus_pos, radius=0.01 * Physics.AU,
                         color=(255, 128, 128), mass=2e24,
                         initial_velocity=[0, -46870], parent=sun)
    p3 = CelestialObject(name="earth", pos=[0.2 * Physics.AU, 0], radius=0.022 * Physics.AU,
                         color=(128, 128, 255), mass=3e21,
                         initial_velocity=[0, 65870], parent=sun)
    p4 = CelestialObject(name="p4", pos=[0 * Physics.AU, 0.3 * Physics.AU], radius=0.016 * Physics.AU,
                         color=(128, 200, 255), mass=3e20,
                         initial_velocity=[-35870, 25000], parent=sun)


def config2():
    sun_pos = [0.5 * Physics.AU, 0.5 * Physics.AU]
    sun = CelestialObject(name="Sun", pos=sun_pos, radius=0.04 * Physics.AU, color=(255, 255, 0), mass=1.989e30,
                          include_in_calcs=False)
    # sun.draw_radius = 40
    # print(sun.pos)
    # print(sun.normalized_pos)

    # moon = CelestialObject(name="Moon", pos=[0.002603 * Physics.AU, 0], radius=1737,
    #                        color=(255, 255, 255), mass=7.34767309e22,
    #                        initial_velocity=[0, 1022], parent=earth)

    mercury_pos = [0.45 * Physics.AU, 0]
    r = np.linalg.norm(mercury_pos)
    mercury_vel = [0, 49870]
    phi = -5 / 7 * np.pi
    x_new = r * np.cos(phi)
    y_new = r * np.sin(phi)
    mercury_pos = [x_new, y_new]

    omega = np.linalg.norm(mercury_vel) / r
    vx_new = -omega * r * np.sin(phi)
    vy_new = omega * r * np.cos(phi)
    mercury_vel = [vx_new, vy_new]

    venus_pos = [-0.3 * Physics.AU, 0]
    p1 = CelestialObject(name="p1", pos=mercury_pos, radius=0.014 * Physics.AU,
                         color=(128, 128, 128), mass=3e23,
                         initial_velocity=mercury_vel, parent=sun)
    p2 = CelestialObject(name="p2", pos=venus_pos, radius=0.01 * Physics.AU,
                         color=(255, 128, 128), mass=2e24,
                         initial_velocity=[0, -46870], parent=sun)
    p3 = CelestialObject(name="earth", pos=[0.2 * Physics.AU, 0], radius=0.022 * Physics.AU,
                         color=(128, 128, 255), mass=3e21,
                         initial_velocity=[0, 65870], parent=sun)
    p4 = CelestialObject(name="p4", pos=[0 * Physics.AU, 0.3 * Physics.AU], radius=0.016 * Physics.AU,
                         color=(128, 200, 255), mass=3e20,
                         initial_velocity=[-35870, 25000], parent=sun)


def initialize_objects():
    for x in Physics.objects:
        x: 'CelestialObject'
        del x
    config1()
    # for obj in Physics.objects:
    #     obj: 'CelestialObject'
    #     obj.perform_calc = True
