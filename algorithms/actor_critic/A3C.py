import multiprocessing as mp
import os
from types import SimpleNamespace as SN

import gym
import matplotlib.pyplot as plt
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import yaml

"""
Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu:
Asynchronous Methods for Deep Reinforcement Learning. ICML 2016: 1928-1937


todo:
GPU

"""


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
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


class RolloutBuffer:
    def __init__(self, args):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.masks = []
        self.args = args

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
                                 nn.ReLU(),
                                 nn.Linear(args.hidden_dim, args.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(args.hidden_dim, args.action_dim))

    def forward(self, state):
        action_pred = self.net(state)
        action_prob = F.softmax(action_pred, dim=-1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(args.state_dim, args.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(args.hidden_dim, args.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(args.hidden_dim, 1))

    def forward(self, state):
        return self.net(state)


class WorkerAgent(mp.Process):
    def __init__(self, args, global_actor, global_critic, global_actor_optimizer, global_critic_optimizer, global_episode, global_reward, global_reward_queue, index):
        super(WorkerAgent, self).__init__()
        self.args = args
        self.name = f'worker-{index}'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(args).to(self.device)
        self.critic = Critic(args).to(self.device)

        self.global_actor = global_actor
        self.global_critic = global_critic
        self.global_actor_optimizer = global_actor_optimizer
        self.global_critic_optimizer = global_critic_optimizer

        self.global_episode = global_episode
        self.global_reward = global_reward
        self.global_reward_queue = global_reward_queue

        self.env = gym.make(args.env_name)
        self.env.seed(index)
        self.buffer = RolloutBuffer(args)

    def run(self):
        while self.global_episode.value < self.args.max_episode:
            state = self.env.reset()
            episode_reward = 0
            while True:
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.rewards.append(reward)
                self.buffer.next_states.append(next_state)
                self.buffer.masks.append(1 - int(done))

                state = next_state
                if done:
                    masks = torch.as_tensor(self.buffer.masks, dtype=torch.float32).to(self.device)
                    rewards = torch.as_tensor(self.buffer.rewards, dtype=torch.float32).to(self.device)
                    # no work in CartPole-v1
                    if self.args.standardize_rewards:
                        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

                    returns = self.calculate_returns(rewards, masks).to(self.device)

                    states = torch.as_tensor(self.buffer.states, dtype=torch.float32).to(self.device)
                    next_states = torch.as_tensor(self.buffer.next_states, dtype=torch.float32).to(self.device)
                    actions = torch.as_tensor(self.buffer.actions, dtype=torch.int64).to(self.device)

                    # GAE calculation
                    with torch.no_grad():
                        state_values = self.critic(states).squeeze().to(self.device)
                        next_state_values = self.critic(next_states).squeeze().to(self.device)
                    advantages = self.calculate_gae(rewards, state_values, next_state_values, masks).to(self.device)

                    logprobs, action_entropy = self.evaluate_action(states, actions)
                    state_values = self.critic(states).squeeze()
                    actor_loss = -(logprobs * advantages).mean()
                    critic_loss = F.mse_loss(state_values, returns)

                    actor_loss.backward()
                    critic_loss.backward()

                    # todo lock
                    # lock = mp.Lock()
                    self.global_actor_optimizer.zero_grad()
                    self.global_critic_optimizer.zero_grad()
                    for param, global_param in zip(self.actor.parameters(), self.global_actor.parameters()):
                        global_param.grad = param.grad
                    for param, global_param in zip(self.critic.parameters(), self.global_critic.parameters()):
                        global_param.grad = param.grad
                    self.global_actor_optimizer.step()
                    self.global_critic_optimizer.step()
                    # lock.release()

                    self.actor.load_state_dict(self.global_actor.state_dict())
                    self.critic.load_state_dict(self.global_critic.state_dict())
                    self.buffer.clear()

                    with self.global_episode.get_lock():
                        self.global_episode.value += 1
                    with self.global_reward.get_lock():
                        if self.global_reward.value == 0.0:
                            self.global_reward.value = episode_reward
                        else:
                            self.global_reward.value = self.global_reward.value * 0.9 + episode_reward * 0.1
                        self.global_reward_queue.put(self.global_reward.value)
                        print(f'{self.name} episode: {self.global_episode.value}  episode reward: {self.global_reward.value}')
                    break

        self.global_reward_queue.put(None)

    @torch.no_grad()
    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        action_prob = self.actor(state)

        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        return action.item()

    # training
    def evaluate_action(self, states, actions):
        action_prob = self.actor(states)
        dist = distributions.Categorical(action_prob)
        action_log_probs = dist.log_prob(actions)
        action_entropy = dist.entropy()
        return action_log_probs, action_entropy

    def calculate_gae(self, rewards, state_values, next_state_values, masks):
        advantage = 0
        advantages = []
        for reward, value, next_value, mask in zip(reversed(rewards), reversed(state_values), reversed(next_state_values), reversed(masks)):
            # TD-error
            delta = reward + self.args.gamma * next_value * mask - value
            advantage = delta + self.args.gamma * self.args.lambda_ * advantage * mask
            advantages.append(advantage)
        advantages.reverse()
        advantages = torch.as_tensor(advantages, dtype=torch.float32)
        return advantages

    def calculate_returns(self, rewards, masks):
        returns = []
        return_ = 0
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            return_ = reward + self.args.gamma * return_ * mask
            returns.append(return_)
        returns.reverse()
        returns = torch.as_tensor(returns, dtype=torch.float32)
        return returns


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
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.show()


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), "config", "actor_critic", "A3C.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
    args = SN(**config_dict)

    train_env = gym.make(args.env_name)
    args.state_dim = train_env.observation_space.shape[0]
    args.action_dim = train_env.action_space.n

    agent = MasterAgent(args)
    agent.train()

    global_actor = Actor(args)
    global_critic = Critic(args)

    global_actor.share_memory()
    global_critic.share_memory()

    global_actor_optimizer = SharedAdam(global_actor.parameters(), lr=1e-4, betas=(0.92, 0.999))
    global_critic_optimizer = SharedAdam(global_critic.parameters(), lr=1e-4, betas=(0.92, 0.999))
    global_episode = mp.Value('i', 0)
    global_reward = mp.Value('d', 0.0)
    global_reward_queue = mp.Queue()

    workers = []
    for i in range(mp.cpu_count()):
        worker = WorkerAgent(args,
                             global_actor,
                             global_critic,
                             global_actor_optimizer,
                             global_critic_optimizer,
                             global_episode,
                             global_reward,
                             global_reward_queue,
                             i)
        workers.append(worker)

    for worker in workers:
        worker.start()

    rewards = []
    while True:
        reward = global_reward_queue.get()
        if reward is None:
            break
        else:
            rewards.append(reward)

    for worker in workers:
        worker.join()

    plt.plot(rewards)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
