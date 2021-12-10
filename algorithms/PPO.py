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
        dist = distributions.Categorical(action_prob)
        return dist


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
        self.actor = Actor(args)
        self.critic = Critic(args)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.buffer = RolloutBuffer(args)

    # inference
    @torch.no_grad()
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        dist = self.actor(state)
        action = dist.sample()
        return action.item()

    # training
    def evaluate_action(self, states, actions):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        dist = self.actor(states)
        action_log_probs = dist.log_prob(actions)
        return action_log_probs

    def train(self):
        # returns calculation
        returns = self.returns_calculation(self.buffer.rewards)

        # PPO
        old_log_probs = self.evaluate_action(self.buffer.states, self.buffer.actions)
        states = torch.tensor(self.buffer.states, dtype=torch.float32)
        for i in range(self.args.K_epoch):
            # GAE calculation
            state_values = self.critic(states).squeeze()
            advantages = self.GAE_calculation(torch.tensor(self.buffer.rewards), state_values)

            new_log_probs = self.evaluate_action(self.buffer.states, self.buffer.actions)
            ratio = (new_log_probs - old_log_probs.detach()).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.args.clip, 1 + self.args.clip) * advantages

            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values, returns)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.buffer.clear()

    def returns_calculation(self, rewards):
        returns = []
        return_ = 0
        for reward in reversed(rewards):
            return_ = reward + return_ * self.args.gamma
            returns.append(return_)
        returns.reverse()
        returns = torch.tensor(returns)
        if self.args.normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        return returns

    def GAE_calculation(self, rewards, state_values):
        next_value = 0
        advantage = 0
        advantages = []
        for r, v in zip(reversed(rewards), reversed(state_values)):
            td_error = r + next_value * self.args.gamma - v
            advantage = td_error + advantage * self.args.gamma * self.args.lambda_
            next_value = v
            advantages.append(advantage)
        advantages.reverse()
        advantages = torch.tensor(advantages)
        if self.args.normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        return advantages

    def save(self, check_point_path):
        pass

    def load(self, check_point_path):
        pass
