import logging
import random
import typing

# import typing
# from typing import Optional, Tuple, Union, List

# import gym
# import pygame
import numpy as np
# from gym.core import ObsType, RenderFrame, ActType

# from src.agent import Agent
# from src.celestial_object import CelestialObject
from src import Physics
# from src.objects import initialize_objects

from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import matplotlib.pyplot as plt
# from IPython.display import display, clear_output
from src.satellite_env import SatelliteEnvironment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
logging.info(device)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, size=128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, size)
        self.layer2 = nn.Linear(size, size)
        self.layer3 = nn.Linear(size, n_actions)

    def forward(self, x):
        x = f.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        return self.layer3(x)


class ReplayMemory(object):

    def __init__(self, capacity, experience):
        self.memory = deque([], maxlen=capacity)
        self.experience = experience

    def push(self, *args):
        self.memory.append(self.experience(*args))

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
            np.exp(-1. * current_step / self.decay)


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

            return torch.tensor(np.array([[action]])).to(device)
        else:
            with torch.no_grad():
                action = policy_net(state).squeeze(1).max(1).indices.view(1, 1)
                # print(f'dqn: {action}')
                return action


class SatelliteTrainer:
    def __init__(self, target, num_episodes=500, batch_size=128, gamma=0.99, eps_start=0.99,
                 eps_end=0.05, eps_decay=1000, tau=0.005, lr=1e-4, memory_size=10000, size=128):
        global device
        self.size = size
        self.target = target
        self.device = device
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.memory_size = memory_size
        self.rewards = []
        self.losses = []
        self.env = SatelliteEnvironment(target=self.target)
        self.n_actions = self.env.action_space.n  # type: ignore
        self.n_observations = self.env.get_current_state()[0].shape[1]
        print('input size:', self.n_observations)
        self.policy_net = DQN(self.n_observations, self.n_actions, size=self.size).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions, size=self.size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr)
        self.experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))
        self.memory = ReplayMemory(self.memory_size, self.experience)
        self.strategy = EpsilonGreedyStrategy(self.eps_start, self.eps_end, self.eps_decay)
        self.agent = Action(self.strategy, self.n_actions, self.env)

        # self.Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))

    def extract_tensors(self, experiences):
        states, actions, rewards, next_states = zip(*experiences)
        t1 = torch.cat(states).to(self.device)
        t2 = torch.cat(actions).to(self.device)
        t3 = torch.cat(rewards).to(self.device)
        t4 = torch.cat(next_states).to(self.device)
        return t1, t2, t3, t4

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        batch = self.experience(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # print("shape", action_batch.shape) # shape torch.Size([128, 1])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(
            #     1).values  # self.target_net(non_final_next_states).squeeze(1).max(1).values ???
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).squeeze(1).max(1).values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def train(self, checkpoint: typing.Optional[str] = None, render=True):
        from data import Data
        if checkpoint:
            checkpoint_load = torch.load(checkpoint)
            self.policy_net.load_state_dict(checkpoint_load['model_state_dict'])
            try:
                self.target_net.load_state_dict(checkpoint_load['target_net_state_dict'])
            except KeyError:
                self.target_net.load_state_dict(checkpoint_load['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint_load['optimizer_state_dict'])
            self.agent.current_step = checkpoint_load['current_step']
            self.policy_net.train()
            print('ok')
        loss = None
        rewards_ep = []
        rewards_sum = []
        losses = []
        data = Data(rewards_ep, rewards_sum, losses)
        for episode in range(self.num_episodes):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)
            cumulative_reward = 0.0
            episode_rewards = []
            for _ in count():
                # print(state.shape)
                action = self.agent.select_action(state, self.policy_net)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device, dtype=torch.float32)
                cumulative_reward += reward.item()
                episode_rewards.append(reward)
                done = terminated or truncated
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device)

                self.memory.push(state, action, next_state, reward)
                state = next_state

                loss = self.optimize_model()
                self.losses.append(loss)

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                            1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    break
                if render:
                    self.env.render()
            rewards_ep.append(episode_rewards)
            self.rewards.append(cumulative_reward)
            data.rewards_ep = rewards_ep
            data.losses = self.losses
            data.rewards_sum = self.rewards
            self.env.episode += 1
            # if episode % 10 == 0:
            #     print(f'ep {episode}')
            if (episode + 1) % 100 == 0:
                # print("save")
                torch.save({
                    'epoch': episode,
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'target_net_state_dict': self.target_net.state_dict(),
                    'loss': loss,
                    'current_step': self.agent.current_step
                }, f'models/checkpoint_2.pth')

        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'current_step': self.agent.current_step
        }, f'models/model.pth')
        data.rewards_ep = rewards_ep
        data.losses = self.losses
        data.rewards_sum = self.rewards
        import pickle
        with open('data.pkl', 'wb') as fn:
            pickle.dump(data, fn)
        # torch.save(self.target_net.state_dict(), f'models/model_final.pth')
        self.plot_results()
        self.env.close()

    def test(self, path='models/model.pth'):
        for obj in Physics.objects:
            del obj
        model = DQN(self.n_observations, self.n_actions, size=self.size).to(self.device)
        data = torch.load(path)
        model.load_state_dict(data['model_state_dict'])
        model.eval()

        env = SatelliteEnvironment(target=self.target)
        n_episodes = 1000
        rewards = []

        for episode in range(n_episodes):

            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            cumulative_reward = 0.0

            for _ in count():
                with torch.no_grad():
                    action = model(state).argmax(dim=1).view(1, 1).to(self.device)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
                cumulative_reward += reward.item()

                done = terminated or truncated
                if done:
                    break

                state = torch.tensor(observation, dtype=torch.float32, device=self.device)
                env.render()

            rewards.append(cumulative_reward)
            env.episode += 1
            # radius = np.random.uniform(0.12, 0.38) * Physics.AU
            # target = 0.5 * Physics.AU + radius * np.array(
            #     [np.cos(theta := np.random.uniform(0, 2 * np.pi)), np.sin(theta)])
            # env.target = target
            # env.target_org = target

        # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # ax.plot(rewards, label='Reward')
        # ax.set_xlabel('Episode')
        # ax.set_ylabel('Reward')
        # ax.set_title('Cumulative Reward per Episode')
        # ax.legend()
        # ax.grid(True)
        #
        # plt.show()
        env.close()

    def plot_results(self, losses=None, rewards=None):
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        if losses is None:
            losses = self.losses
        if rewards is None:
            rewards = self.rewards
        axs[0].plot(losses, label='Loss')
        axs[0].set_xlabel('Iterations')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Training Loss')
        axs[0].legend()
        axs[0].grid(True)
        axs[1].plot(rewards, label='Cumulative Reward')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Cumulative Reward')
        axs[1].set_title('Cumulative Reward over Episodes')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()


# def main():
#     target = [0.18 * Physics.AU, 0.294 * Physics.AU]
#     radius = np.random.uniform(0.12, 0.38) * Physics.AU
#     target = 0.5 * Physics.AU + radius * np.array(
#         [np.cos(theta := np.random.uniform(0, 2 * np.pi)), np.sin(theta)])
#
#     target = [0.18 * Physics.AU, 0.294 * Physics.AU]
#     # target = [0.31 * Physics.AU, 0.81 * Physics.AU]
#     trainer = SatelliteTrainer(target, num_episodes=600)
#     # trainer.train(render=False)  # , checkpoint='models/model_final.pth')
#     trainer.test('models/model_final.pth')
#
#
# if __name__ == '__main__':
#     main()
    # random_mov()
    # main()
