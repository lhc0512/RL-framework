import gym
from ppo import PPOAgent

class Parameters:
    def __init__(self):
        self.total_steps = 500000
        self.train_steps = 2048
        self.test_steps = 5000
        self.save_steps = 100000
        self.max_episode_len = 400
        self.file_id = '2020-12-06-001'
        self.K_epochs = 10
        self.clip = 0.2
        self.lr_actor = 0.0003
        self.lr_critic = 0.001
        self.hidden_dim = 64
        self.gamma = 0.99
        self.seed = 123456
        self.env_name = 'CartPole-v1'
        # self.visible_GPU = '0'


def test_agent(env, agent):
    episode_reward = 0
    state = test_env.reset()
    done = False
    while not done:
        action, logprob = agent.select_action(state)
        state, reward, done, info = env.step(action)
        episode_reward += reward
    print(episode_reward)
    # return episode_reward


if __name__ == '__main__':
    args = Parameters()
    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)

    args.state_dim = env.observation_space.shape[0]  # (4,)
    args.action_dim = env.action_space.n
    # args.max_episode_steps = env._max_episode_steps

    agent = PPOAgent(args)

    current_steps = 0
    while current_steps < args.total_steps:
        state = env.reset()
        for i in range(1, args.max_episode_len + 1):
            action, logprob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            # buffer
            agent.buffer.states.append(state)
            agent.buffer.actions.append(action)
            agent.buffer.rewards.append(reward)
            agent.buffer.done.append(done)
            agent.buffer.logprobs.append(logprob)

            current_steps += 1
            if done:
                state = env.reset()
            else:
                state = next_state

            if current_steps % args.train_steps == 0:
                agent.train()

            if current_steps % args.test_steps == 0:
                test_agent(test_env, agent)

            if current_steps % args.save_steps == 0:
                agent.save(args.file_id + '/' + str(current_steps))
