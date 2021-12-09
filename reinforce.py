import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions


class RolloutBuffer:
    def __init__(self, args):
        self.rewards = []
        self.logprobs = []
        self.args = args

    def put(self, reward, logprob):
        self.rewards.append(reward)
        self.logprobs.append(logprob)

    def clear(self):
        self.rewards.clear()
        self.logprobs.clear()


class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.net = nn.Sequential(nn.Linear(args.state_dim, args.hidden_dim),
                                 # todo Tanh
                                 # nn.Tanh(),
                                 nn.ReLU(),
                                 # todo Dropout
                                 # nn.Dropout(),
                                 nn.Linear(args.hidden_dim, args.hidden_dim),
                                 # nn.Tanh(),
                                 nn.ReLU(),
                                 nn.Linear(args.hidden_dim, args.hidden_dim),
                                 # todo Hardswish
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

    def select_action(self, state):
        # 添加一维度
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # gradient
        action_pred = self.policy(state)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        lob_prob_action = dist.log_prob(action)
        return action.item(), lob_prob_action

    def train(self):
        # reward to go
        returns = []
        return_ = 0
        for reward in reversed(self.buffer.rewards):
            return_ = reward + return_ * self.args.gamma
            returns.insert(0, return_)
        returns = torch.tensor(returns, dtype=torch.float32)
        # normalization
        if self.args.normalize:
            returns = (returns - returns.mean()) / returns.std()
        logprobs = torch.stack(self.buffer.logprobs).squeeze()
        loss = - (returns * logprobs).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.buffer.clear()

    def save(self, check_point_path):
        pass

    def load(self, check_point_path):
        pass
