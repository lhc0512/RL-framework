import gym
import numpy as np
import torch

from algorithms.PPO import ActorCriticAgent
from utils import plotLearning
import os
import yaml
from types import SimpleNamespace as SN
from datetime import datetime
from multiprocessing import Pipe, Process

with open(os.path.join(os.path.dirname(__file__), "config", "PPO.yaml"), "r") as f:
    try:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        assert False, "default.yaml error: {}".format(exc)
args = SN(**config_dict)


def test_agent(env, agent):
    episode_reward = 0
    for i in range(args.test_num):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_argmax_action(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
    return episode_reward / args.test_num


if __name__ == '__main__':
    train_env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)

    train_env.seed(args.seed)
    test_env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.state_dim = train_env.observation_space.shape[0]
    args.action_dim = train_env.action_space.n

    agent = ActorCriticAgent(args)
    episode_reward_history = []
    current_steps = 0

    while current_steps < args.total_steps:
        state = train_env.reset()
        done = False
        while not done:
            current_steps += 1
            action = agent.select_action(state)
            next_state, reward, done, info = train_env.step(action)
            # buffer
            agent.buffer.states.append(state)
            agent.buffer.actions.append(action)
            agent.buffer.rewards.append(reward)
            agent.buffer.next_states.append(next_state)
            agent.buffer.masks.append(1 - int(done))

            state = next_state

            if current_steps % args.train_steps == 0:
                agent.train()
            if current_steps % args.test_steps == 0:
                average_episode_reward = test_agent(test_env, agent)
                episode_reward_history.append(average_episode_reward)
                print(f'| step : {current_steps:6} | Episode Reward: {average_episode_reward:5.1f} |')
            if current_steps % args.save_steps == 0:
                agent.save(f'{args.checkpoint_path}/{args.seed}-{args.name}-{current_steps}')


    file_name = f'{args.seed}-{args.name}-{args.env_name}-{datetime.now()}.png'
    plotLearning(episode_reward_history, filename=file_name, window=25)
