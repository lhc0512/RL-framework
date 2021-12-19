"""
based on
https://github.com/MorvanZhou/pytorch-A3C

Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu:
Asynchronous Methods for Deep Reinforcement Learning. ICML 2016: 1928-1937

Asynchronous Advantage Actor Critic (A3C)
"""

import os

import gym
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "1"

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.masks = []

    def add(self, state, action, reward, next_state, mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.masks.append(mask)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.masks.clear()


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(args.state_dim, args.hidden_dim),
                                 nn.Tanh(),
                                 nn.Linear(args.hidden_dim, args.hidden_dim),
                                 nn.Tanh(),
                                 nn.Linear(args.hidden_dim, args.action_dim))

    def forward(self, state):
        action_pred = self.net(state)
        action_prob = F.softmax(action_pred, dim=-1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(args.state_dim, args.hidden_dim),
                                 nn.Tanh(),
                                 nn.Linear(args.hidden_dim, args.hidden_dim),
                                 nn.Tanh(),
                                 nn.Linear(args.hidden_dim, 1))

    def forward(self, state):
        return self.net(state).squeeze()


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class WorkerAgent(mp.Process):
    def __init__(self, args, global_actor, global_critic, global_actor_optimizer, global_critic_optimizer, global_ep, global_ep_r, res_queue, name):
        super(WorkerAgent, self).__init__()
        self.name = f'worker-{name}'
        self.global_episode, self.global_episode_reward, self.res_queue = global_ep, global_ep_r, res_queue
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.global_actor_optimizer = global_actor_optimizer
        self.global_critic_optimizer = global_critic_optimizer
        self.actor = Actor(args)
        self.critic = Critic(args)
        self.env = gym.make('CartPole-v1')
        self.buffer = RolloutBuffer()
        self.args = args

    @torch.no_grad()
    def choose_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        prob = self.actor(state)
        dist = Categorical(prob)
        return dist.sample().item()

    def evaluate_action(self, states, actions):
        probs = self.actor(states)
        dist = Categorical(probs)
        logprobs = dist.log_prob(actions)
        return logprobs

    def run(self):
        total_step = 1
        while self.global_episode.value < self.args.max_episodes:
            state = self.env.reset()
            episode_reward = 0.
            while True:
                action = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                self.buffer.add(state, action, reward, next_state, (1 - int(done)))
                # update global and assign to local net
                if total_step % self.args.update_steps == 0 or done:
                    # next state value
                    with torch.no_grad():
                        next_state_value = self.critic(torch.as_tensor(next_state, dtype=torch.float32)).detach().numpy() * (1 - int(done))
                    buffer_v_target = []
                    for reward in reversed(self.buffer.rewards):  # reverse buffer r
                        next_state_value = reward + self.args.gamma * next_state_value
                        buffer_v_target.append(next_state_value)
                    buffer_v_target.reverse()

                    v_target = torch.as_tensor(buffer_v_target, dtype=torch.float32)
                    states = torch.as_tensor(self.buffer.states, dtype=torch.float32)
                    actions = torch.as_tensor(self.buffer.actions, dtype=torch.int32)

                    logprobs = self.evaluate_action(states, actions)
                    values = self.critic(states)
                    # todo warning: 1 element vs 1 element array []
                    critic_loss = F.mse_loss(v_target, values)
                    actor_loss = (-logprobs * (v_target - values).detach()).mean()
                    # !!!!
                    # actor_loss.backward()
                    # critic_loss.backward()
                    # calculate local gradients and push local parameters to global
                    self.global_actor_optimizer.zero_grad()
                    self.global_critic_optimizer.zero_grad()
                    # zero_grad -> backward -> step
                    # todo  5 hours !!!!
                    actor_loss.backward()
                    critic_loss.backward()
                    for actor_param, global_actor_param in zip(self.actor.parameters(), self.global_actor.parameters()):
                        global_actor_param.grad = actor_param.grad
                    for critic_param, global_critic_param in zip(self.critic.parameters(), self.global_critic.parameters()):
                        global_critic_param.grad = critic_param.grad
                    self.global_actor_optimizer.step()
                    self.global_critic_optimizer.step()

                    # pull global parameters
                    self.actor.load_state_dict(self.global_actor.state_dict())
                    self.critic.load_state_dict(self.global_critic.state_dict())
                    self.buffer.clear()

                    # done and print information
                    if done:
                        with self.global_episode.get_lock():
                            self.global_episode.value += 1
                        with self.global_episode_reward.get_lock():
                            if self.global_episode_reward.value == 0.:
                                self.global_episode_reward.value = episode_reward
                            else:
                                self.global_episode_reward.value = self.global_episode_reward.value * 0.9 + episode_reward * 0.1
                        self.res_queue.put(self.global_episode_reward.value)
                        print(f'episode: {self.global_episode.value} | reward: {self.global_episode_reward.value}')
                        break
                state = next_state
                total_step += 1
        self.res_queue.put(None)


class MasterAgent:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_actor = Actor(args).to(self.device)
        self.global_critic = Critic(args).to(self.device)

        self.global_actor.share_memory()
        self.global_critic.share_memory()

        self.global_actor_optimizer = torch.optim.Adam(self.global_actor.parameters(), lr=args.actor_lr)
        self.global_critic_optimizer = torch.optim.Adam(self.global_critic.parameters(), lr=args.critic_lr)
        self.global_episode = mp.Value('i', 0)
        self.global_reward = mp.Value('d', 0.0)
        self.global_reward_queue = mp.Queue()
        self.args = args

    def train(self):
        workers = []
        for i in range(mp.cpu_count()):
            worker = WorkerAgent(self.args,
                                 self.global_actor,
                                 self.global_critic,
                                 self.global_actor_optimizer,
                                 self.global_critic_optimizer,
                                 self.global_episode,
                                 self.global_reward,
                                 self.global_reward_queue,
                                 i)
            workers.append(worker)

        for worker in workers:
            worker.start()

        rewards = []
        while True:
            reward = self.global_reward_queue.get()
            if reward is None:
                break
            else:
                rewards.append(reward)

        for worker in workers:
            worker.join()

        plt.plot(rewards)
        plt.ylabel('reward')
        plt.xlabel('step')
        file_name = f'{self.args.seed}-{self.args.name}-{self.args.env_name}-{datetime.now()}.png'
        plt.savefig(file_name, format="png")
        plt.close()
