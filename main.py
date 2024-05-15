import logging
import random
import typing
from typing import Optional, Tuple, Union, List

import gym
import pygame
import numpy as np
from gym.core import ObsType, RenderFrame, ActType

from src.agent import Agent
from src.celestial_object import CelestialObject
from src import Physics
from src.objects import initialize_objects

from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
logging.info(device)


# import tensorflow as tf


def render(satellite):
    pygame.init()
    screen_width, screen_height = 800, 600
    window = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Systemy inteligentne")
    # clock = pygame.time.Clock()
    running = True
    scaling_factor = 1.0
    panning = False
    panning_offset = np.array([0, 0])
    panning_start = np.array([0, 0])
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

            satellite.handle_events()  # sterowanie

        forces = {}
        for obj in CelestialObject.objects:
            force = obj.calculate_total_force(CelestialObject.objects)
            forces[obj] = force

        for obj, force in forces.items():
            obj.update_position(*force)
            obj.draw(window, panning_offset[0], panning_offset[1], True, (screen_width, screen_height), scaling_factor)

        # satellite.draw_info(window, (screen_width, screen_height))
        # satellite.draw(window, panning_offset[0], panning_offset[1], True, (screen_width, screen_height),
        #                scaling_factor)

        velocity_text = font.render(
            f"Velocity: {satellite.velocity[0] / 1000:.2f}, {satellite.velocity[1] / 1000:.2f} km/s", True,
            (255, 255, 255))
        window.blit(velocity_text, (10, 10))
        pygame.display.flip()
        # clock.tick(5)


def main_render():
    initialize_objects()

    # sun_pos = [400 / Physics.SCALE, 300 / Physics.SCALE]
    # sun = CelestialObject(name="Sun", pos=sun_pos, radius=30, color=(255, 255, 0), mass=1.989e30)
    #
    # earth = CelestialObject(name="Earth", pos=[1 * Physics.AU, 0], radius=3.7,
    #                         color=(0, 0, 255), mass=5.972e24,
    #                         initial_velocity=[0, 29784], parent=sun)
    #
    # moon = CelestialObject(name="Moon", pos=[0.002603 * Physics.AU, 0], radius=1,
    #                        color=(255, 255, 255), mass=7.34767309e22,
    #                        initial_velocity=[0, 1022], parent=earth)
    objects = Physics.objects
    earth = next((obj for obj in objects if obj.name.lower() == 'earth'), None)
    satellite = Agent(name="Satellite", pos=[0.002 * Physics.AU, 0], radius=0.0068, color=(255, 128, 255), mass=1000,
                      initial_velocity=[0, 1022], parent=earth, fuel_capacity=10000)
    render(satellite)


class SatelliteEnvironment(gym.Env):
    def __init__(self, target=None):
        num_celestial_objects = 11
        observation_size = num_celestial_objects + 2 + 2 + 2 + 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_size,),
                                                dtype=np.float32)

        initialize_objects()
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) # moze discrete
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.integer)
        earth = next((obj for obj in Physics.objects if obj.name.lower() == 'earth'), None)
        self.satellite = Agent(name="Satellite", pos=[0.002 * Physics.AU, 0], radius=0.0068, color=(255, 128, 255),
                               mass=1000,  # moze 6kh
                               initial_velocity=[0, 1022], parent=earth, fuel_capacity=10000, target=target)
        self.screen = None
        self.panning = False
        self.panning_offset = 0, 0
        self.panning_start = [0, 0]
        self.scaling_factor = 1
        self.tol = 2137  # km

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        Physics.objects = []
        initialize_objects()
        earth = next((obj for obj in Physics.objects if obj.name.lower() == 'earth'), None)
        self.satellite = Agent(name="Satellite", pos=[0.002 * Physics.AU, 0], radius=0.0068, color=(255, 128, 255),
                               mass=1000,  # moze masa z 6kg
                               initial_velocity=[0, 1022], parent=earth, fuel_capacity=10000)
        # self.screen = None
        self.panning = False
        self.panning_offset = 0, 0
        self.panning_start = [0, 0]
        self.scaling_factor = 1
        distances = self.satellite.distances()
        state = [distances]
        state.extend(self.get_current_state())
        padded_state = [np.pad(arr, (0, 12 - len(arr)), mode='constant') for arr in state]
        # (observation, reward, terminated, truncated, info)
        data = {
            'distances': distances,
            'truncated': False,
            'terminated': False,
            'reward': 0,
            'collision': False,
            'fuel': self.satellite.fuel_level
        }
        return padded_state, data  # todo zmienic

    def render(self, mode='human'):
        screen_width, screen_height = 800, 600
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.panning = True
                    self.panning_start = np.array(pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.panning = False
            elif event.type == pygame.MOUSEMOTION:
                if self.panning:
                    current_pos = np.array(pygame.mouse.get_pos())
                    self.panning_offset += current_pos - self.panning_start
                    self.panning_start = current_pos
            elif event.type == pygame.MOUSEWHEEL:
                self.scaling_factor *= 1.25 if event.y > 0 else 0.75
                self.scaling_factor = max(0.75, self.scaling_factor)
        for obj in CelestialObject.objects:
            obj.draw(self.screen, self.panning_offset[0], self.panning_offset[1], True, (screen_width, screen_height),
                     self.scaling_factor)
        pygame.display.flip()
        return pygame.surfarray.array3d(self.screen)

    def get_current_state(self) -> typing.List:
        pos = self.satellite.pos
        fuel = self.satellite.fuel_level
        fx, fy, _ = self.satellite.calculate_total_force(self.satellite.celestial_objects)
        return [self.satellite.target, pos, np.array([fuel]), np.array([fx, fy]), self.satellite.velocity]

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        self.satellite.set_thrust_direction([action[0], action[1]])
        forces = {}
        for obj in CelestialObject.objects:
            force = obj.calculate_total_force(CelestialObject.objects)
            forces[obj] = force
        for obj, force in forces.items():
            fx, fy, _ = force
            obj.update_position(fx, fy)
        # for obj in CelestialObject.objects:
        #     force = obj.calculate_total_force(CelestialObject.objects)
        #     forces[obj] = force
        #     obj.update_position(*force)
        """
        ObsSpace: array - dystans do obiektow; (11)
        xy targetu; (2)
        obecny xy; (2)
        sila wypadkowa; (2)
        poziom paliwa. (1)
        """
        distances = self.satellite.distances()
        r, dist = self.reward()  # reward
        d = dist < self.tol * 1000  # done
        collision = self.satellite.collision(distances)
        t = collision or self.satellite.fuel_level <= 0
        state = [distances]
        state.extend(self.get_current_state())
        padded_state = [np.pad(arr, (0, 12 - len(arr)), mode='constant') for arr in state]
        # (observation, reward, terminated, truncated, info)
        data = {
            'distances': distances,
            'truncated': t,
            'terminated': d,
            'reward': r,
            'collision': collision,
            'fuel': self.satellite.fuel_level
        }
        # return [6, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], r, d, t, data
        return padded_state, r, d, t, data

    def reward(self):
        # todo....
        # funkcja odleglosci potencjalnej orbity od dystansu?????
        distance_to_target = self.satellite.distance_to_target()
        fuel_consumed = self.satellite.fuel_capacity - self.satellite.fuel_level
        distance_reward = 1 / distance_to_target if distance_to_target > 0 else 0
        fuel_penalty = fuel_consumed * 0
        out_of_fuel_penalty = -100 if self.satellite.fuel_level <= 0 else 0
        total_reward = distance_reward - fuel_penalty + out_of_fuel_penalty
        return total_reward, distance_to_target


def random_mov():
    done = False
    sun_pos = [400 / Physics.SCALE, 300 / Physics.SCALE]
    target = np.add([2.53 * Physics.AU, 0], sun_pos)
    env = SatelliteEnvironment(target=target)
    for i in range(100000000):
        if done:
            env.reset()
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        # print(env.action_space.sample())
        env.render()
    env.close()


############################
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = f.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        return self.layer3(x)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            np.exp(-1. * current_step * self.decay)


class Action:
    def __init__(self, strategy, num_actions, env):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.env = env

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = self.env.action_space.sample()
            return torch.tensor([[action]]).to(device)
        else:
            with torch.no_grad():
                action = policy_net(state).max(1).indices.view(-1, 1, 2)
                print(action)
                return action


def main():
    sun_pos = [400 / Physics.SCALE, 300 / Physics.SCALE]
    target = np.add([2.53 * Physics.AU, 0], sun_pos)
    env = SatelliteEnvironment(target=target)
    num_episodes = 500
    n_actions = 2
    n_observations = 12
    batch_size = 128
    gamma = 0.99
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 1000
    tau = 0.005
    lr = 1e-4
    memory_size = 10000
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(memory_size)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Action(strategy, n_actions, env)
    Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))

    def extract_tensors(experiences):
        # Unpack the tuple of experiences
        states, actions, rewards, next_states = zip(*experiences)
        t1 = torch.cat(states).to(device)
        t2 = torch.cat(actions).to(device)
        t3 = torch.cat(rewards).to(device)
        t4 = torch.cat(next_states).to(device)

        return t1, t2, t3, t4

    # Training loop
    for episode in range(num_episodes):
        state, data = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        done = False
        episode_reward = 0
        print('ep1')
        while not done:
            # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = agent.select_action(state, policy_net)
            print(action)
            # print(f'action: {action[0]}')
            next_state, reward, done, _, __ = env.step(action[0][0].cpu().numpy())
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            # state, r, d, t, data
            # print(reward)
            episode_reward += reward
            reward = torch.tensor([reward], device=device)
            # print(action)
            # print(state.shape)
            memory.push((state, action, reward, next_state))
            state = next_state

            if len(memory) > batch_size:
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)

                # Check the shapes of the states and actions tensors
                print("Shapes of states and actions tensors:")
                print("States:", states.shape)
                print("Actions:", actions.shape)

                current_q_values = policy_net(states).gather(dim=1, index=actions)
                next_q_values = target_net(next_states).max(dim=1)[0].detach()
                target_q_values = rewards + gamma * next_q_values

                loss = f.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # env.render()
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}: Total Reward: {episode_reward}")

    env.close()


if __name__ == '__main__':
    main()
    # random_mov()
# main()

