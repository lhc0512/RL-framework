import gym
import numpy as np
import torch

from reinforce import ReinforceAgent
from utils import plotLearning


class Arguments:
    def __init__(self):
        self.env_name = 'CartPole-v1'
        self.file_id = '2020-12-06'
        self.seed = 1234
        self.state_dim = None
        # todo 32 64 128
        self.hidden_dim = 128
        self.action_dim = None
        self.lr = 0.001
        self.max_episode = 300
        self.gamma = 0.99
        self.test_episode_interval = 10
        self.normalize = True


if __name__ == '__main__':
    args = Arguments()
    train_env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    train_env.seed(args.seed)
    test_env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.state_dim = train_env.observation_space.shape[0]
    args.action_dim = train_env.action_space.n

    agent = ReinforceAgent(args)
    episode_reward_history = []
    for episode in range(1, args.max_episode + 1):
        state = train_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, log_prob_action = agent.select_action(state)
            next_state, reward, done, info = train_env.step(action)
            agent.buffer.put(reward, log_prob_action)
            state = next_state
            episode_reward += reward
        episode_reward_history.append(episode_reward)
        agent.train()
        if episode % args.test_episode_interval == 0:
            print(f'| Episode: {episode:3} | Episode Reward: {episode_reward:5.1f} |')
    filename = 'CartPole-v1.png'
    plotLearning(episode_reward_history, filename=filename, window=25)
