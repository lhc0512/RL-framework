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


class ActorCriticAgent:
    def __init__(self, args):
        self.args = args
        self.actor_critic = ActorCritic(args)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=args.lr)
        self.buffer = RolloutBuffer(args)

    # inference
    @torch.no_grad()
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        action_prob, state_value = self.actor_critic(state)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        return action.item()

    # training
    def act(self, states, actions):
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions)
        action_prob, state_values = self.actor_critic(states)
        dist = distributions.Categorical(action_prob)
        log_prob_actions = dist.log_prob(actions)
        return state_values.squeeze(), log_prob_actions

    def train(self):
        # ----------- returns -------------------------------------
        returns = []
        return_ = 0
        for reward in reversed(self.buffer.rewards):
            return_ = reward + return_ * self.args.gamma
            returns.append(return_)
        returns.reverse()
        returns = torch.tensor(returns)
        if self.args.normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        # ---------------------------------------------------------

        values, logprobs = self.act(self.buffer.states, self.buffer.actions)
        advantage = returns - values.detach()
        actor_loss = -(logprobs * advantage).mean()
        # todo critic_loss
        # critic_loss = F.smooth_l1_loss(values, returns).mean()
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.buffer.clear()

    def save(self, check_point_path):
        pass

    def load(self, check_point_path):
        pass
