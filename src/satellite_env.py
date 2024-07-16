import typing
from typing import Optional, Tuple

import gym
import pygame
import numpy as np
from IPython.core.display_functions import clear_output
from gym.core import ObsType, ActType

import matplotlib.pyplot as plt

from src import CelestialObject
from src.physics import Physics
from src.agent import Agent
from src.objects import initialize_objects


class SatelliteEnvironment(gym.Env):
    def __init__(self, target=None):
        num_celestial_objects = 11
        self.target = np.array(target)
        observation_size = num_celestial_objects + 2 + 2 + 2 + 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_size,),
                                                dtype=np.float32)
        initialize_objects()
        self.action_space = gym.spaces.Discrete(9, start=0)
        self.earth = next((obj for obj in Physics.objects if obj.name.lower() == 'earth'), None)
        self.sun = next((obj for obj in Physics.objects if obj.name.lower() == 'sun'), None)
        self.sun_pos = self.sun.pos
        self.satellite = self.init_satellite()
        self.target_org = np.copy(target)
        self.last_pos = self.satellite.pos
        self.target_org = target
        self.reached = False
        self.rewards = []
        self.episodes = []
        self.previous = []
        self.screen = None
        self.episode = 0
        self.tol = 0.03 * Physics.AU
        self.next_orbit = np.full(128, np.inf, dtype=float)
        self.losses = []
        self.action_map: typing.Dict = {
            8: [1, -1],
            1: [1, 0],
            2: [1, 1],
            3: [0, 1],
            4: [-1, 0],
            5: [-1, -1],
            6: [0, -1],
            7: [-1, 1],
            0: [0, 0]
        }

        # self.action_map: typing.Dict = {
        #     0: [0, 0],
        #     1: [-1, 0],
        #     2: [0, 1],
        #     3: [0, -1],
        #     4: [1, 0]
        # }

        self.closest = np.inf

    def init_satellite(self):
        satellite = Agent(name="Satellite", pos=[0.05 * Physics.AU, +0.1*Physics.AU], radius=1, color=(255, 128, 255), mass=1,
                          initial_velocity=[0, -105870], parent=self.earth, fuel_capacity=3000,
                          target=self.target)
        return satellite

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        Physics.objects = []
        initialize_objects()
        if self.satellite is not None:
            del self.satellite
        self.earth = next((obj for obj in Physics.objects if obj.name.lower() == 'earth'), None)
        self.sun = next((obj for obj in Physics.objects if obj.name.lower() == 'sun'), None)
        self.satellite = self.init_satellite()
        self.reached = False
        self.satellite.reached = False
        ###
        self.satellite.target_points = self.target
        self.satellite.target = self.target
        self.next_orbit = np.full(128, np.inf, dtype=float)
        self.last_pos = self.satellite.pos
        self.rewards = []

        state, state_data = self.get_current_state()
        distances = state_data['distances']
        data = {
            'distances': distances,
            'truncated': False,
            'terminated': False,
            'reward': 0,
            'collision': False,
            'fuel': self.satellite.fuel_level,
            'closest': self.closest
        }
        return state, data

    def render(self, mode='human'):
        screen_width, screen_height = Physics.W, Physics.H
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.screen.fill((0, 0, 0))
        pygame.event.pump()
        for obj in CelestialObject.objects:
            obj.draw(self.screen)
        pygame.display.flip()
        return pygame.surfarray.array3d(self.screen)

    def get_current_state(self) -> typing.Tuple[np.ndarray, typing.Dict]:
        pos = self.satellite.pos
        fuel = self.satellite.fuel_level
        rotation = self.satellite.orientation
        mass = self.satellite.mass
        fx, fy, _ = self.satellite.calculate_total_force(self.satellite.celestial_objects)
        distances = self.satellite.distances()
        dist_to_target = self.satellite.distance_to_target()
        lds = self.satellite.lidars
        data = {
            'pos': pos / Physics.AU,  # / Physics.AU,
            'target': self.satellite.target / Physics.AU,  # / Physics.AU,
            'orientation': np.array([self.satellite.orientation]),
            'angle_vector': self.satellite.angle_vector(),
            'min_dist': np.array([np.min(distances)]) / Physics.AU,
            # 'positions': np.array([obj.pos for obj in self.satellite.celestial_objects]) / Physics.AU,
            # 'thrust_direction': np.array(self.satellite.thrust_direction),
            'velocity': self.satellite.velocity / np.linalg.norm(self.satellite.velocity),
            'planet_ld': np.array(lds),
            'target_ld': self.satellite.calc_l_vector(self.satellite.target)
        }
        data_raw = {
            'pos': pos,
            'dist_to_target': np.array([dist_to_target]),
            'target': np.array(self.satellite.target),
            'closest': np.array([self.closest]),
            'total_forces': np.array([fx, fy]),
            'fuel': np.array([fuel]),
            'thrust_direction': np.array([self.satellite.thrust_direction]),
            'thrust_force': np.array([self.satellite.thrust_force]),
            'rotation': np.array([rotation]),
            'mass': np.array([mass]),
            'distances': distances,
            # 'obj_pos': np.array([x.pos for x in self.satellite.celestial_objects]),
        }
        state = list(data.values())
        r_state = [arr.reshape(1, -1) for arr in state]
        c_state = np.concatenate(r_state, axis=1)
        # print(c_state.shape)
        return c_state, data_raw

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # print(action)
        self.last_pos = self.satellite.pos
        mapped_action = self.action_map[action]
        self.satellite.set_thrust_direction(mapped_action)
        n_steps = 12
        for i in range(n_steps):
            forces = {}
            for obj in CelestialObject.objects:
                force = obj.calculate_total_force(CelestialObject.objects)
                forces[obj] = force
            for obj, force in forces.items():
                fx, fy, _ = force
                obj.update_position(fx, fy)

        state, state_data = self.get_current_state()
        distances = state_data['distances']
        # predicted_orbit = self.satellite.simulate_next(512)
        r, dist, d, t = self.reward(obj_distances=distances)  # reward
        # if len(self.rewards) > 9000: t = True
        # d = dist < self.tol * 1000
        # collision = self.satellite.collision(distances)
        # t = collision or self.satellite.fuel_level <= 0 or #(dist / Physics.AU > 25)
        data = {
            'distances': distances,
            'truncated': d,
            'terminated': t,
            'reward': r,
            'fuel': self.satellite.fuel_level,
            'distance_to_target': dist
        }
        self.rewards.append(r)
        if self.screen is not None:
            text = [f'reward: {r:.2f}', f'sum_rewards: {np.sum(self.rewards):.2f}', f'iter: {len(self.rewards)}',
                    f'episode: {self.episode}']
            self.satellite.text_list = text
        self.satellite.reached = self.reached
        # self.satellite.draw_text(self.screen, text)
        # return [6, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], r, d, t, data
        return state, r, t, d, data

    @staticmethod
    def concatenate_state(state):
        c_state = np.concatenate(state, axis=1)
        return c_state

    def reward(self, obj_distances):
        last_dist = np.linalg.norm(self.target - self.last_pos)
        current_dist = np.linalg.norm(self.target - self.satellite.pos)
        goal_vel_reward = (last_dist - current_dist) / Physics.AU * 100
        # if current_dist > 0.4:
        #     if 0 < goal_vel_reward < 0.04:
        #         goal_vel_reward = 0
        sum_safety = 0
        ps, rad = zip(*map(lambda x: (x.pos, x.radius), self.satellite.celestial_objects))
        prev_dist = np.min(np.linalg.norm(np.array(ps) - np.array(self.last_pos), axis=1) - np.array(rad))
        safety_factor = 100
        mindist = np.min(obj_distances)
        if mindist < 0.1 * Physics.AU:
            if prev_dist >= mindist:
                sum_safety -= safety_factor * (prev_dist - mindist) / Physics.AU
        d = False
        sum_safety2 = 0
        scaled_pos = self.satellite.pos / Physics.AU
        if scaled_pos[0] < 0.1:
            if self.last_pos[0] / Physics.AU >= scaled_pos[0]:
                sum_safety2 -= safety_factor * (self.last_pos[0] - self.satellite.pos[0]) / Physics.AU
        if scaled_pos[1] < 0.1:
            if self.last_pos[1] / Physics.AU >= scaled_pos[1]:
                sum_safety2 -= safety_factor * (self.last_pos[1] - self.satellite.pos[1]) / Physics.AU
        if scaled_pos[0] > 0.9:
            if self.last_pos[0] / Physics.AU <= scaled_pos[0]:
                sum_safety2 += safety_factor * (self.last_pos[0] - self.satellite.pos[0]) / Physics.AU
        if scaled_pos[1] > 0.9:
            if self.last_pos[1] / Physics.AU <= scaled_pos[1]:
                sum_safety2 += safety_factor * (self.last_pos[1] - self.satellite.pos[1]) / Physics.AU
        reward = (
                0.1 +
                + 5 * np.arctan(goal_vel_reward)
                + 12 * np.arctan(sum_safety)
                + 12 * np.arctan(sum_safety2)
        )  # + 1 / (0.33 + mindist / Physics.AU)
        if current_dist < self.tol:
            # if not self.reached:
            #     print('osiagnieto')
            self.satellite.score += 1
            self.reached = True
            reward += 6
        out = np.any(scaled_pos > 1) or np.any(scaled_pos < 0)
        t = self.satellite.collision(obj_distances) or out
        if t and self.reached:
            # radius = np.random.uniform(0.12, 0.38) * Physics.AU
            # self.target = 0.5 * Physics.AU + radius * np.array(
            #     [np.cos(theta := np.random.uniform(0, 2 * np.pi)), np.sin(theta)])
            # self.satellite.target = self.target
            self.reached = False
        d = self.satellite.score > 200
        return reward, current_dist, d, t

    def plot_rewards(self):
        clear_output(wait=True)
        plt.figure(figsize=(10, 5))
        plt.plot(self.episodes, self.rewards, label='Cumulative Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
