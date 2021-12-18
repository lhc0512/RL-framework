import os
from types import SimpleNamespace as SN

import gym
import yaml

from algorithms.actor_critic.shareA3C import MasterAgent

with open(os.path.join(os.path.dirname(__file__), "config", "actor_critic", "A3C.yaml"), "r") as f:
    try:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        assert False, "default.yaml error: {}".format(exc)
args = SN(**config_dict)

if __name__ == '__main__':
    train_env = gym.make(args.env_name)
    args.state_dim = train_env.observation_space.shape[0]
    args.action_dim = train_env.action_space.n

    agent = MasterAgent(args)
    agent.train()
