from multiprocessing import Pipe, Process
from copy import deepcopy
import gym


def env_worker(conn, env):
    while True:
        method, parameters = conn.recv()
        if method == 'reset':
            env.reset()
            state = env.reset()
            conn.send(state)

        elif method == 'step':
            action = parameters
            next_state, reward, done, info = env.step(action)
            conn.send((next_state, reward, done, info))

        elif method == 'close':
            env.close()
            conn.close()
            break


class ParallelRunner:
    def __init__(self, args):
        self.args = args
        self.master_conns = []
        self.worker_conns = []

        for _ in range(self.args.agent_nums):
            master_conn, worker_conn = Pipe()
            self.master_conns.append(master_conn)
            self.worker_conns.append(worker_conn)

        env_args = []
        for i in range(self.args.agent_nums):
            copy_args = deepcopy(args)
            copy_args.seed += i
            env_args.append(copy_args)

        self.multi_process = []
        for env_arg, worker_conn in zip(env_args, self.worker_conns):
            env = gym.make(args.env_name)
            self.multi_process.append(Process(target=env_worker, args=(worker_conn, env)))

        for process in self.multi_process:
            process.daemon = True
            process.start()

    def close(self):
        for conn in self.master_conns:
            conn.send(('close', None))

    def reset(self):
        for conn in self.master_conns:
            conn.send(('reset', None))
        for conn in self.master_conns:
            state = conn.recv()
