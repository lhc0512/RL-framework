import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np


class RolloutBuffer:
    def __init__(self, args):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.logprobs = []
        self.args = args

    def put(self, state, action, reward, next_state, done, logprob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.logprobs.append(logprob)

    def get_batches(self):
        # rewards = []
        # discounted_reward = 0
        # for reward, done in zip(reversed(self.rewards), self.dones):
        #     if done:
        #         discounted_reward = 0
        #     discounted_reward = reward + self.args.gamma * discounted_reward
        #     rewards.append(discounted_reward)
        # rewards.reverse()
        # self.rewards = rewards
        n = len(self.states)
        # batch_start_index = np.arange(0, n, self.args.batch_size)
        indices = np.arange(n, dtype=np.int64)
        np.random.shuffle(indices)
        # batches = []
        # for i in batch_start_index:
        #     batches.append(indices[i:i + self.args.batch_size])
        return torch.tensor(self.states, dtype=torch.float32), torch.tensor(self.actions), torch.tensor(self.rewards), torch.tensor(self.next_states ,dtype=torch.float32), torch.tensor(self.dones), torch.tensor(
            self.logprobs), indices

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.logprobs.clear()


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                 nn.ReLU(),
                                 # nn.Tanh(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 # nn.Tanh(),
                                 # todo Hardswish
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.Hardswish(),
                                 nn.Linear(hidden_dim, action_dim),
                                 # todo remove
                                 nn.Softmax(dim=-1)
                                 )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                 nn.ReLU(),
                                 # nn.Tanh(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 # nn.Tanh(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.Hardswish(),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, state):
        return self.net(state)


class PPOAgent:
    def __init__(self, args):
        self.actor = Actor(args.state_dim, args.hidden_dim, args.action_dim)
        self.critic = Critic(args.state_dim, args.hidden_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr_critic)
        self.buffer = RolloutBuffer(args)
        self.args = args

    @torch.no_grad()
    def select_action(self, state):
        state = torch.FloatTensor(state)
        # target
        action_prob = self.actor(state)
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def train(self):
        # todo advantage/reward normalizing
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 参数拟合数据, 数据不变
        for _ in range(self.args.K_epochs):
            states_arr, actions_arr, rewards_arr, next_states_arr, done_arr, logprobs_arr, batch = self.buffer.get_batches()
            """GAE"""
            with torch.no_grad():
                states_value_arr = self.critic(states_arr)
                next_states_value_arr = self.critic(next_states_arr)
                advantages_arr = torch.zeros(len(states_arr)+1)
                for t in reversed(range(len(states_arr))):
                    delta = rewards_arr[t] + self.args.gamma * next_states_value_arr[t] * (1-done_arr[t]) - states_value_arr[t]
                    advantages_arr[t] = delta + self.args.gamma * self.args.lambda_ * advantages_arr[t+1] * (1-done_arr[t])
                advantages_arr = advantages_arr[:len(states_arr)]

            # for batch in batches:
            states = states_arr[batch]
            old_logprobs = logprobs_arr[batch]
            old_actions = actions_arr[batch]
            rewards = rewards_arr[batch]
            advantages = advantages_arr[batch]

            state_values = self.critic(states)
            dist = Categorical(self.actor(states))
            logprobs = dist.log_prob(old_actions)
            # todo detach
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.args.clip, 1 + self.args.clip) * advantages
            # actor loss
            actor_loss = - torch.min(surr1, surr2).mean()
            # critic loss
            critic_loss = 0.5 * (state_values - rewards) ** 2
            critic_loss = critic_loss.mean()

            total_loss = actor_loss + critic_loss
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.buffer.clear()

    # def save(self, check_point_path):
    #     torch.save(self.actor.state_dict(), check_point_path + 'actor')
    #     torch.save(self.critic.state_dict(), check_point_path + 'critic')
    #
    # def load(self, check_point_path):
    #     self.actor.load_state_dict(torch.load(check_point_path + 'actor'))
    #     self.critic.load_state_dict(torch.load(check_point_path + 'critic'))
