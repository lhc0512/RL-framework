import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

"""
Advantage Actor Critic (A2C)
"""


class RolloutBuffer:
    def __init__(self, args):
        self.states = []
        self.actions = []
        self.rewards = []
        self.args = args

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(args.state_dim, args.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(args.hidden_dim, args.hidden_dim),
                                 nn.Hardswish(),
                                 nn.Linear(args.hidden_dim, args.action_dim))

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(args.state_dim, args.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(args.hidden_dim, args.hidden_dim),
                                 nn.Hardswish(),
                                 nn.Linear(args.hidden_dim, 1))

    def forward(self, state):
        return self.net(state)


class ActorCriticAgent:
    def __init__(self, args):
        self.args = args
        self.actor = Actor(args)
        self.critic = Critic(args)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.buffer = RolloutBuffer(args)

    # inference
    @torch.no_grad()
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_pred = self.actor(state)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        return action.item()

    # training
    def act(self, states, actions):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)

        action_preds = self.actor(states)
        action_probs = F.softmax(action_preds, dim=-1)
        dist = distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)

        values = self.critic(states)

        return action_log_probs, values.squeeze()

    def train(self):
        """returns"""
        returns = []
        return_ = 0
        for reward in reversed(self.buffer.rewards):
            return_ = reward + return_ * self.args.gamma
            returns.append(return_)
        returns.reverse()
        returns = torch.tensor(returns)
        if self.args.normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        logprobs, values = self.act(self.buffer.states, self.buffer.actions)
        advantage = returns - values.detach()
        actor_loss = -(logprobs * advantage).mean()
        critic_loss = F.mse_loss(values, returns)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.buffer.clear()

    def save(self, check_point_path):
        pass

    def load(self, check_point_path):
        pass
