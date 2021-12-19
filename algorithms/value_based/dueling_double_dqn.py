import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

"""
Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas:
Dueling Network Architectures for Deep Reinforcement Learning. ICML 2016: 1995-2003
"""


class ReplayBuffer:
    def __init__(self, args):
        self.states = np.zeros((args.buffer_size, args.state_dim), dtype=np.float32)
        self.actions = np.zeros((args.buffer_size, 1), dtype=np.int32)
        self.rewards = np.zeros((args.buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros((args.buffer_size, args.state_dim), dtype=np.float32)
        self.masks = np.zeros((args.buffer_size, 1), dtype=np.int32)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.index = 0
        self.size = 0

    def put(self, state, action, reward, next_state, mask):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.masks[self.index] = mask

        self.index = (self.index + 1) % self.args.buffer_size
        self.size = min(self.size + 1, self.args.buffer_size)

    def sample(self):
        batch = np.random.randint(0, self.size, size=self.args.batch_size)
        states = torch.as_tensor(self.states[batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.actions[batch]).long().to(self.device)
        rewards = torch.as_tensor(self.rewards[batch], dtype=torch.float32).to(self.device)
        next_states = torch.as_tensor(self.next_states[batch], dtype=torch.float32).to(self.device)
        masks = torch.as_tensor(self.masks[batch], dtype=torch.int32).to(self.device)
        return states, actions, rewards, next_states, masks


class DuelingDoubleDQN(nn.Module):
    def __init__(self, args):
        super(DuelingDoubleDQN, self).__init__()
        self.net = nn.Sequential(nn.Linear(args.state_dim, args.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(args.hidden_dim, args.hidden_dim),
                                 nn.ReLU())
        self.value = nn.Linear(args.hidden_dim, 1)
        self.advantage = nn.Linear(args.hidden_dim, args.action_dim)

    def forward(self, state):
        x = self.net(state)
        value = self.value(x)
        advantage = self.advantage(x)
        q_value = torch.add(value, advantage - advantage.mean(-1, keepdim=True))
        return q_value


class DuelingDoubleDQNAgent:
    def __init__(self, args):
        self.q_net = DuelingDoubleDQN(args)
        self.q_target = copy.deepcopy(self.q_net)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.buffer = ReplayBuffer(args)
        self.args = args

    @torch.no_grad()
    def select_action(self, state):
        if np.random.random() > self.args.epsilon:
            state = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1).to(self.device)
            action = torch.argmax(self.q_net(state)).item()
        else:
            action = np.random.randint(0, self.args.action_dim)
        return action

    # test
    @torch.no_grad()
    def select_argmax_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1).to(self.device)
        return torch.argmax(self.q_net(state)).item()

    def train(self):
        if self.buffer.size < self.args.batch_size:
            return
        states, actions, rewards, next_states, masks = self.buffer.sample()
        with torch.no_grad():
            # double DQN
            argmax_actions = self.q_net(next_states).argmax(-1).unsqueeze(-1)
            y_i = rewards + self.args.gamma * self.q_target(next_states).gather(-1, argmax_actions) * masks

        q_values = self.q_net(states)
        q_values = q_values.gather(-1, actions)
        loss = F.mse_loss(y_i, q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # todo soft update & hard update
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        # todo epsilon update
        self.args.epsilon = self.args.epsilon * self.args.epsilon_decay if self.args.epsilon > self.args.epsilon_mini else self.args.epsilon_mini

    def save(self, checkpoint_path):
        torch.save(self.q_net.state_dict(), f'{checkpoint_path}.pth')

    def load(self, checkpoint_path):
        self.q_net.load_state_dict(torch.load(f'{checkpoint_path}.pth'))
