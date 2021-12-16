import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np

"""
todo list:
vec env
distributed
GPU
"""


class RolloutBuffer:
    def __init__(self, args):
        self.states = []
        self.avail_actions = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.masks = []
        self.args = args

    def clear(self):
        self.states.clear()
        self.avail_actions.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.masks.clear()

    def get_batches(self):
        n = len(self.states)
        indices = np.arange(n, dtype=np.int64)
        batch_start = np.arange(0, n, self.args.batch_size)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.args.batch_size] for i in batch_start]
        return batches


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(args.state_dim, args.hidden_dim),
                                 nn.Tanh(),
                                 # nn.ReLU(),
                                 nn.Linear(args.hidden_dim, args.hidden_dim),
                                 nn.Tanh(),
                                 # nn.Hardswish(),
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
                                 # nn.Hardswish(),
                                 nn.ReLU(),
                                 nn.Linear(args.hidden_dim, 1))

    def forward(self, state):
        return self.net(state)


class ActorCriticAgent:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(args).to(self.device)
        self.critic = Critic(args).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.buffer = RolloutBuffer(args)

    # inference in generating data
    @torch.no_grad()
    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        action_prob = self.actor(state)

        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        return action.item()

    # test
    @torch.no_grad()
    def select_argmax_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        action_prob = self.actor(state)

        return torch.argmax(action_prob).item()

    # training
    def evaluate_action(self, states, actions):
        action_prob = self.actor(states)
        dist = distributions.Categorical(action_prob)
        action_log_probs = dist.log_prob(actions)
        action_entropy = dist.entropy()

        return action_log_probs, action_entropy

    def train(self):
        # tensor buffer
        masks = torch.as_tensor(self.buffer.masks, dtype=torch.float32).to(self.device)

        rewards = torch.as_tensor(self.buffer.rewards, dtype=torch.float32).to(self.device)
        # no work in CartPole-v1
        if self.args.standardize_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        returns = self.calculate_returns(rewards, masks)

        states = torch.as_tensor(self.buffer.states, dtype=torch.float32).to(self.device)
        next_states = torch.as_tensor(self.buffer.next_states, dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(self.buffer.actions, dtype=torch.int64).to(self.device)

        # GAE calculation
        with torch.no_grad():
            old_state_values = self.critic(states).squeeze().to(self.device)
            old_next_state_values = self.critic(next_states).squeeze().to(self.device)
        advantages = self.calculate_GAE(rewards, old_state_values, old_next_state_values, masks).to(self.device)

        # critic loss target
        if self.args.use_returns:  # true
            targets = returns
        else:
            targets = advantages + old_state_values
        # PPO
        with torch.no_grad():
            old_log_probs, _ = self.evaluate_action(states, actions)

        for i in range(self.args.K_epoch):
            batches = self.buffer.get_batches()
            batches = torch.as_tensor(batches, dtype=torch.int64).to(self.device)
            for batch in batches:
                new_log_probs, prob_entropy = self.evaluate_action(states[batch], actions[batch])
                ratio = (new_log_probs - old_log_probs[batch]).exp()
                surr1 = ratio * advantages[batch]
                surr2 = ratio.clamp(1 - self.args.clip, 1 + self.args.clip) * advantages[batch]
                actor_loss = - torch.min(surr1, surr2).mean()

                # entropy
                if self.args.use_entropy:  # false
                    actor_loss += - self.args.entropy_coeffient * prob_entropy.mean()

                new_state_values = self.critic(states[batch]).squeeze()
                # value clip
                if self.args.use_clipped_value:  # false
                    critic_loss1 = 0.5 * (old_state_values[batch] + torch.clamp(new_state_values - old_state_values[batch], - self.args.clip, self.args.clip) - targets[batch]) ** 2
                    critic_loss2 = 0.5 * (new_state_values - targets[batch]) ** 2
                    critic_loss = torch.max(critic_loss1, critic_loss2).mean()
                else:
                    critic_loss = F.mse_loss(new_state_values, targets[batch])

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()

                # gradient clip
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.clip_grad_norm)

                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.buffer.clear()

    def calculate_returns(self, rewards, masks):
        returns = []
        return_ = 0
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            return_ = reward + self.args.gamma * return_ * mask
            returns.append(return_)
        returns.reverse()
        returns = torch.as_tensor(returns, dtype=torch.float32)
        return returns

    def calculate_GAE(self, rewards, state_values, next_state_values, masks):
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

    def save(self, checkpoint_path):
        torch.save(self.actor.state_dict(), f'{checkpoint_path}_actor.pth')
        torch.save(self.critic.state_dict(), f'{checkpoint_path}_critic.pth')

    def load(self, checkpoint_path):
        self.actor.load_state_dict(torch.load(f'{checkpoint_path}_actor.pth'))
        self.critic.load_state_dict(torch.load(f'{checkpoint_path}_critic.pth'))
