from torch import nn
import torch
from copy import deepcopy
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 # todo Hardswish
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.Hardswish(),
                                 nn.Linear(hidden_dim, action_dim),
                                 # todo remove
                                 nn.Softmax(dim=-1)
                                 )

    def forward(self, state):
        return self.net(state)

    def evaluate(self, state, action):
        action_probs = self.net(state)
        dist = Categorical(action_probs)
        action_lobprobs = dist.log_prob(action)
        return action_lobprobs


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.Hardswish(),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, state):
        return self.net(state)


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.done = []
        # todo
        self.logprobs = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.done.clear()
        self.logprobs.clear()


class PPOAgent:
    def __init__(self, args):
        self.actor = Actor(args.state_dim, args.hidden_dim, args.action_dim)
        self.critic = Critic(args.state_dim, args.hidden_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr_critic)

        self.actor_old = deepcopy(self.actor)
        self.critic_old = deepcopy(self.critic)
        self.buffer = RolloutBuffer()
        self.MSE = nn.MSELoss()
        self.args = args

    @torch.no_grad()
    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_prob = self.actor(state)
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def train(self):
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.buffer.rewards), self.buffer.done):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.args.gamma * discounted_reward
            rewards.append(discounted_reward)
        rewards.reverse()
        rewards = torch.tensor(rewards, dtype=torch.float32)
        # todo advantage/reward normalizing
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        states = torch.FloatTensor(self.buffer.states)
        old_actions = torch.tensor(self.buffer.actions)
        # todo
        old_logprobs = torch.tensor(self.buffer.logprobs)

        for _ in range(self.args.K_epochs):
            logprobs = self.actor.evaluate(states, old_actions)
            state_values = self.critic(states)
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.args.clip, 1 + self.args.clip) * advantages
            loss = - torch.min(surr1, surr2) + 0.5 * self.MSE(state_values, rewards)
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.mean().backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.buffer.clear()

    def save(self, check_point_path):
        torch.save(self.actor_old.state_dict(), check_point_path + 'actor')
        torch.save(self.critic.state_dict(), check_point_path + 'critic')

    def load(self, check_point_path):
        self.actor.load_state_dict(torch.load(check_point_path + 'actor'))
        self.critic.load_state_dict(torch.load(check_point_path + 'critic'))
