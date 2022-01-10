import os
from types import SimpleNamespace as SN

import numpy as np
import yaml
from numpy import random

with open(os.path.join(os.path.dirname(__file__), "../", "config", "envs", "edge_computing.yaml"), "r") as f:
    try:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        assert False, "default.yaml error: {}".format(exc)
args = SN(**config_dict)

np.random.seed(args.seed)
random.seed(args.seed)

K = 1024
Byte = 8


class Task:
    def __init__(self, args):
        # Task Tuple
        self.args = args
        self.task_size = np.random.uniform(args.task_size_min, args.task_size_max) * K * Byte  # 影响传输时间
        self.task_cpu_cycle = np.random.uniform(args.task_complexity_min, args.task_complexity_max) * self.task_size  # 影响计算时间

        self.task_deadline = args.deadline  # 任务截止时间

        self.transmission_waiting_time = 0  # 传输等待时间
        self.transmission_time = 0  # 任务需要的传输时间
        self.current_transmission_time = 0  # 当前传输花费时间

        self.execute_waiting_time = 0  # 执行等待时间
        self.execute_time = 0  # 任务需要的执行时间
        self.current_execute_time = 0  # 当前执行时间

        self.cost = 0
        # todo 记录可靠性数据
        self.alpha = 0
        self.beta = 0
        self.reliability = 0
