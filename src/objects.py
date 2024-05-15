from src.physics import Physics
from src.celestial_object import CelestialObject


def initialize_objects():
    sun_pos = [400 / Physics.SCALE, 300 / Physics.SCALE]
    sun = CelestialObject(name="Sun", pos=sun_pos, radius=696340, color=(255, 255, 0), mass=1.989e30)

    earth = CelestialObject(name="Earth", pos=[1 * Physics.AU, 0], radius=6378,
                            color=(0, 0, 255), mass=5.972e24,
                            initial_velocity=[0, 29784], parent=sun)

    moon = CelestialObject(name="Moon", pos=[0.002603 * Physics.AU, 0], radius=1737,
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
    mercury = CelestialObject(name="Mercury", pos=mercury_pos, radius=2440,
                              color=(128, 128, 128), mass=3.3011e23,
                              initial_velocity=[0, 47870], parent=sun)
    venus = CelestialObject(name="Venus", pos=venus_pos, radius=6052,
                            color=(255, 215, 0), mass=4.8675e24,
                            initial_velocity=[0, 35020], parent=sun)
    mars = CelestialObject(name="Mars", pos=mars_pos, radius=3390,
                           color=(255, 0, 0), mass=6.417e23,
                           initial_velocity=[0, 24130], parent=sun)

    jupiter_initial_velocity = [0, 13070]  # m/s
    io_initial_velocity = [0, 17000]
    io_pos = [0.00282089577 * Physics.AU, 0]
    jupiter = CelestialObject(name="Jupiter", pos=jupiter_pos, radius=6991,
                              color=(255, 165, 0), mass=1.898e27,
                              initial_velocity=jupiter_initial_velocity, parent=sun)

    # io = CelestialObject(name="Io", pos=io_pos, radius=1821,
    #                      color=(200, 200, 200), mass=8.931938e22,
    #                      initial_velocity=io_initial_velocity, parent=jupiter)
    saturn = CelestialObject(name="Saturn", pos=saturn_pos, radius=58232,
                             color=(255, 215, 0), mass=5.683e26,
                             initial_velocity=[0, 9690], parent=sun)
    uranus = CelestialObject(name="Uranus", pos=uranus_pos, radius=25362,
                             color=(173, 216, 230), mass=8.681e25,
                             initial_velocity=[0, 6810], parent=sun)
    neptune = CelestialObject(name="Neptune", pos=neptune_pos, radius=24622,
                              color=(30, 144, 255), mass=1.024e26,
                              initial_velocity=[0, 5430], parent=sun)
    pluto = CelestialObject(name="Pluto", pos=pluto_pos, radius=1188,
                            color=(192, 192, 192), mass=1.303e22,
                            initial_velocity=[0, 4749], parent=sun)
