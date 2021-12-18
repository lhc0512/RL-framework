import multiprocessing as mp

import gym
import matplotlib.pyplot as plt
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F

"""
Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu:
Asynchronous Methods for Deep Reinforcement Learning. ICML 2016: 1928-1937


todo:
GPU

"""


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


class ActorCritic(nn.Module):
    def __init__(self, args):
        super(ActorCritic, self).__init__()
        # share network
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        # actor
        self.pi = nn.Linear(args.hidden_dim, args.action_dim)
        # critic
        self.value = nn.Linear(args.hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action_prob = F.softmax(self.pi(x), dim=-1)
        state_value = self.value(x)  # []
        return action_prob, state_value


class MasterAgent:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_actor_critic = ActorCritic(args)
        self.global_actor_critic_optimizer = torch.optim.Adam(self.global_actor_critic.parameters(), lr=args.lr)
        self.global_episode = mp.Value('i', 0)
        self.global_reward = mp.Value('d', 0.0)
        self.global_reward_queue = mp.Queue()
        self.args = args

    def train(self):
        workers = []
        for i in range(mp.cpu_count()):
            worker = WorkerAgent(self.args,
                                 self.global_actor_critic,
                                 self.global_actor_critic_optimizer,
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


class WorkerAgent(mp.Process):
    def __init__(self, args, global_actor_critic, global_actor_critic_optimizer, global_episode, global_reward, global_reward_queue, index):
        super(WorkerAgent, self).__init__()
        self.args = args
        self.name = f'worker-{index}'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor_critic = ActorCritic(args)
        self.global_actor_critic = global_actor_critic
        self.global_actor_critic_optimizer = global_actor_critic_optimizer

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
                        _, state_values = self.actor_critic(states)
                        state_values = state_values.squeeze().to(self.device)
                        _, next_state_values = self.actor_critic(next_states)
                        next_state_values = next_state_values.squeeze().to(self.device)

                    advantages = self.calculate_gae(rewards, state_values, next_state_values, masks).to(self.device)

                    logprobs, action_entropy = self.evaluate_action(states, actions)
                    _, state_values = self.actor_critic(states)
                    state_values = state_values.squeeze()
                    actor_loss = -(logprobs * advantages).mean()
                    critic_loss = F.mse_loss(state_values, returns)

                    actor_loss.backward()
                    critic_loss.backward()

                    # todo lock
                    # lock = mp.Lock()
                    self.global_actor_critic_optimizer.zero_grad()
                    for param, global_param in zip(self.actor_critic.parameters(), self.global_actor_critic.parameters()):
                        global_param.grad = param.grad
                    self.global_actor_critic_optimizer.step()
                    # lock.release()

                    self.actor_critic.load_state_dict(self.global_actor_critic.state_dict())
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
        action_prob, _ = self.actor_critic(state)

        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        return action.item()

    # training
    def evaluate_action(self, states, actions):
        action_prob, _ = self.actor_critic(states)
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
