import gym
import numpy as np
import torch
import os
import yaml
from types import SimpleNamespace as SN
from algorithms.policy_gradient.reinforce import ReinforceAgent
from commons.utils import plot_figure

if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), "configs", "policy_gradient", "reinforce.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
    args = SN(**config_dict)

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
            action = agent.select_action(state)
            next_state, reward, done, info = train_env.step(action)
            agent.buffer.states.append(state)
            agent.buffer.actions.append(action)
            agent.buffer.rewards.append(reward)
            state = next_state
            episode_reward += reward
        episode_reward_history.append(episode_reward)
        agent.train()
        if episode % args.test_episode_interval == 0:
            print(f'| Episode: {episode:3} | Episode Reward: {episode_reward:5.1f} |')
    filename = 'CartPole-v1.png'
    plot_figure(episode_reward_history, "Episode","Reward", filename=filename)
