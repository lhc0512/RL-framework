import gym
from ppo import PPOAgent


class Arguments:
    def __init__(self):
        self.total_steps = 100000
        self.train_steps = 2048
        self.test_steps = 5000
        self.save_steps = 100000
        self.max_episode_len = 500
        self.file_id = '2020-12-06-001'
        self.K_epochs = 10
        self.clip = 0.2
        self.lr_actor = 0.0003
        self.lr_critic = 0.001
        self.hidden_dim = 256
        self.gamma = 0.99
        self.lambda_ = 0.95
        self.seed = 123456
        self.env_name = 'CartPole-v1'
        self.batch_size = 512


def test_agent(env, agent):
    episode_reward = 0
    state = test_env.reset()
    done = False
    while not done:
        action, logprob = agent.select_action(state)
        state, reward, done, info = env.step(action)
        episode_reward += reward
    return episode_reward


if __name__ == '__main__':
    args = Arguments()
    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)

    args.state_dim = env.observation_space.shape[0]  # (4,)
    args.action_dim = env.action_space.n

    agent = PPOAgent(args)

    current_steps = 0
    while current_steps < args.total_steps:
        state = env.reset()
        for i in range(1, args.max_episode_len + 1):
            current_steps += 1
            action, logprob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            # buffer
            agent.buffer.put(state, action, reward, next_state, done, logprob)
            state = next_state

            if current_steps % args.train_steps == 0:
                agent.train()

            if current_steps % args.test_steps == 0:
                episode_reward = test_agent(test_env, agent)
                print('steps: {}'.format(current_steps), 'score:', episode_reward)

            # if current_steps % args.save_steps == 0:
            #     agent.save(args.file_id + '/' + str(current_steps))
