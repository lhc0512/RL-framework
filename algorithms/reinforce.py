import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions


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


class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.net = nn.Sequential(nn.Linear(args.state_dim, args.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(args.hidden_dim, args.hidden_dim),
                                 nn.Hardswish(),
                                 nn.Linear(args.hidden_dim, args.action_dim))

    def forward(self, state):
        return self.net(state)


class ReinforceAgent:
    def __init__(self, args):
        self.args = args
        self.policy = Policy(args)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.buffer = RolloutBuffer(args)

    @torch.no_grad()
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_pred = self.policy(state)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        return action.item()

    def act(self, states, actions):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)

        action_pred = self.policy(states)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        return dist.log_prob(actions)

    def train(self):
        # returns
        returns = []
        return_ = 0
        for reward in reversed(self.buffer.rewards):
            return_ = reward + return_ * self.args.gamma
            returns.append(return_)
        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32)
        if self.args.normalize:
            returns = (returns - returns.mean()) / returns.std()

        logprobs = self.act(self.buffer.states, self.buffer.actions)
        loss = - (returns * logprobs).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.buffer.clear()

    def save(self, check_point_path):
        pass

    def load(self, check_point_path):
        pass
