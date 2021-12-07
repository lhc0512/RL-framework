import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym

# environment
env = gym.make('CartPole-v0')
# observation dimension
obs_dim = env.observation_space.shape[0]
# action dimension
act_dim = env.action_space.n

# policy network
net = nn.Sequential(
    nn.Linear(obs_dim, 32), nn.Tanh(),
    nn.Linear(32, act_dim), nn.Identity()
)

# optimizer
optimizer = Adam(net.parameters(), lr=1e-2)

print('\nUsing simplest formulation of policy gradient.\n')

# training loop
for i in range(50):
    # make some empty lists for logging.
    batch_observations = []
    batch_actions = []
    batch_weights = []  # for R(tau) weighting in policy gradient
    batch_returns = []  # for measuring episode returns
    batch_lengths = []  # for measuring episode lengths

    # reset episode-specific variables
    observation = env.reset()  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over
    ep_rewards = []  # list for rewards accrued throughout ep

    # collect experience by acting in the environment with current policy
    while True:
        # save observation
        batch_observations.append(observation.copy())
        # act in the environment
        policy = Categorical(logits=net(torch.as_tensor(observation, dtype=torch.float32)))
        action = policy.sample().item()
        observation, reward, done, info = env.step(action)

        # save action, reward
        batch_actions.append(action)
        ep_rewards.append(reward)

        if done:
            # if episode is over, record info about episode
            ep_return, ep_length = sum(ep_rewards), len(ep_rewards)
            batch_returns.append(ep_return)
            batch_lengths.append(ep_length)

            # the weight for each logprob(a_t|s_t) is reward-to-go from t
            reward_to_gos = np.zeros_like(ep_rewards)
            reward_to_gos[-1] = ep_rewards[-1]
            for k in reversed(range(ep_length-1)):
                reward_to_gos[k] = ep_rewards[k] + reward_to_gos[k + 1]
            batch_weights += list(reward_to_gos)

            # reset episode-specific variables
            observation, done, ep_rewards = env.reset(), False, []

            # end experience loop if we have enough of it
            if len(batch_observations) > 5000:
                break

    # take a single policy gradient update step
    optimizer.zero_grad()
    # make loss function whose gradient, for the right data, is policy gradient
    policy = Categorical(logits=net(torch.as_tensor(batch_observations, dtype=torch.float32)))
    logp = policy.log_prob(torch.as_tensor(batch_actions, dtype=torch.int32))
    batch_loss = -(logp * torch.as_tensor(batch_weights, dtype=torch.float32)).mean()
    batch_loss.backward()
    optimizer.step()
    print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' % (i, batch_loss, np.mean(batch_returns), np.mean(batch_lengths)))
