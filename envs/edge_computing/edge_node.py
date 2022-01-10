import os
from numpy import random
import yaml
from collections import deque
from types import SimpleNamespace as SN
import numpy as np
import math
from envs.edge_computing.task import Task
import copy

with open(os.path.join(os.path.dirname(__file__), "../", "config", "envs", "edge_computing.yaml"), "r") as f:
    try:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        assert False, "default.yaml error: {}".format(exc)
args1 = SN(**config_dict)
# 这里设置随机数, 后面创建的多个edge_node 的配置就是多样的
np.random.seed(args1.seed)
random.seed(args1.seed)

K = 1024
M = 1024 * 1024
G = 1024 * 1024 * 1024
Byte = 8


class EdgeNode:
    def __init__(self, args):
        self.args = args
        self.id = 0
        self.edge_nodes = []
        self.cpu_core_num = np.random.choice(args.cpu_core_list)
        self.cpu_capacity = self.cpu_core_num * G * args.single_core_cpu_capacity
        self.transmission_rates = []
        self.task_probability = np.random.uniform(args.task_probability_min, args.task_probability_max)
        self.x = np.random.randint(0, 1000)  # 坐标x
        self.y = np.random.randint(0, 1000)  # 坐标y
        self.execution_failure_rate = np.random.uniform(args.execution_failure_rate_min, args.execution_failure_rate_max)
        self.transmission_failure_rates = [np.random.uniform(args.transmission_failure_rate_min, args.transmission_failure_rate_max) for _ in range(args.edge_node_num)]  # n
        self.executed_queue = deque()  # 正在在计算的队列 1个
        self.received_queues = [deque() for _ in range(args.edge_node_num)]  # OFDMA N-1 queue  为了coding 方便，初始化为N # n
        self.sent_queues = [deque() for _ in range(args.edge_node_num)]  # OFDMA N-1 queue # n
        self.obs_shape = 6 + args.edge_node_num * 4
        self.executed_queue_len = int(self.cpu_core_num / args.cpu_core_list[-1] * args.max_queue_len)
        self.new_task = None

    def init_edge_node_transmission_rates(self, transmission_rates):
        self.transmission_rates = transmission_rates

    def get_observation(self):
        observation = []
        observation.append(self.execution_failure_rate / self.args.execution_failure_rate_max)
        observation.append(self.cpu_capacity / (self.args.cpu_core_list[-1] * self.args.single_core_cpu_capacity * G))
        observation.append(self.task_probability / self.args.task_probability_max)
        max_transmission_rate = max(self.transmission_rates)
        for v in self.transmission_rates:
            observation.append(v / max_transmission_rate)

        max_transmission_failure_rate = max(self.transmission_failure_rates)
        for v in self.transmission_failure_rates:
            observation.append(v / max_transmission_failure_rate)

        # 归一化 结合 max queue len
        observation.append(len(self.executed_queue) / self.args.max_queue_len)
        for received_queue in self.received_queues:
            observation.append(len(received_queue) / self.args.max_queue_len)
        for sent_queue in self.sent_queues:
            observation.append(len(sent_queue) / self.args.max_queue_len)
        # task observation
        if self.new_task:
            observation.append(self.new_task.task_size / (self.args.task_size_max * K * Byte))
            observation.append(self.new_task.task_cpu_cycle / (self.args.task_complexity_max * self.args.task_size_max * K * Byte))
        else:
            observation.extend([0, 0])
        return observation

    # ! 更合理的IPPO并行训练, 指的是环境是统一的,而产生的任务数据是多样的, 但是, 边缘计算环境下, 总是产生不同的任务, 也不影响训练
    def generate_task(self):
        is_task_arrival = np.random.binomial(1, self.task_probability)
        self.new_task = None
        if is_task_arrival == 1:
            self.new_task = Task(self.args)

    def reset_edge_node(self):
        self.executed_queue.clear()
        for received_queue in self.received_queues:
            received_queue.clear()
        for sent_queue in self.sent_queues:
            sent_queue.clear()

    def get_avail_actions(self):
        if self.new_task:
            avail_actions = [1] * (self.args.edge_node_num + 1)
            avail_actions[self.args.edge_node_num] = 0
            #  训练环境, 可以知道全局信息
            if self.args.reject_task == True:
                # 使用队列长度限制
                for i in range(self.args.edge_node_num):
                    # 进一步设计根据核数, 定制每个节点的最大队列长度
                    if len(self.edge_nodes[i].executed_queue) > self.edge_nodes[i].executed_queue_len:
                        avail_actions[i] = 0
                # 若是没有其他节点可以卸载, 就在本地执行
                avail_actions[self.id] = 1
            return avail_actions
        else:
            # 没有任务，不做处理
            avail_actions = [0] * (self.args.edge_node_num + 1)
            avail_actions[self.args.edge_node_num] = 1
            return avail_actions

    def offload_task(self, action):
        if action == self.args.edge_node_num:
            pass
        else:
            # 对任务执行offloading
            task = self.new_task
            if action == self.id:
                task.transmission_time = 0
                task.execute_time = task.task_cpu_cycle / self.cpu_capacity
                task.cost = task.transmission_time + task.execute_time * self.execution_failure_rate
                task.reliability = math.exp(-task.cost)
                self.executed_queue.append(task)
            else:
                task.transmission_time = task.task_size / self.transmission_rates[action]
                # TODO 解耦
                task.execute_time = task.task_cpu_cycle / self.edge_nodes[action].cpu_capacity
                task.cost = task.transmission_time * self.transmission_failure_rates[action] + task.execute_time * self.edge_nodes[action].execution_failure_rate
                task.reliability = math.exp(-task.cost)
                self.sent_queues[action].append(task)  # 虚设
                self.edge_nodes[action].received_queues[self.id].append(copy.deepcopy(task))  # 实际影响效果

    def execute_task(self):
        task_completion_time = 0
        task_total_cost = 0
        task_execution_waiting_time = 0
        task_execution_time = 0
        task_transmission_waiting_time = 0
        task_transmission_time = 0
        drop_task_number = 0
        finished_task_number = 0
        task_reliability = 0
        # -------------------------------------------
        # 对local queue 的处理 物理上只有一个队列
        # 时间片
        if len(self.executed_queue) > 0:
            task = self.executed_queue[0]
            task.current_execute_time += self.args.mini_time_slot
            for i in range(1, len(self.executed_queue)):
                waiting_task = self.executed_queue[i]
                waiting_task.execute_waiting_time += self.args.mini_time_slot
            finished_or_dropped = False
            task_time = task.execute_waiting_time + task.current_execute_time + task.transmission_waiting_time + task.current_transmission_time
            if task_time > task.task_deadline:
                task_total_cost += self.args.task_penalty
                drop_task_number += 1
                finished_or_dropped = True
            elif task.current_execute_time >= task.execute_time:
                task_total_cost += task.cost
                finished_or_dropped = True
            if finished_or_dropped:
                self.executed_queue.popleft()
                finished_task_number += 1
                # debug information
                task_completion_time += task_time
                task_execution_waiting_time += task.execute_waiting_time
                task_execution_time += task.current_execute_time
                task_transmission_waiting_time += task.transmission_waiting_time
                task_transmission_time += task.current_transmission_time
                task_reliability += task.reliability
        # 检测任务, 如果超过截止时间, 给予惩罚, 并删除队列中的任务
        new_executed_queue = deque()
        for task in self.executed_queue:  # 线程安全, 不可修改的
        # todo 不知道为何, 这种遍历方式就是错 IndexError: deque index out of range
        # for i in range(len(self.executed_queue)):  # 可修改, 会出问题
        #     task = self.executed_queue[i]
            task_time = task.execute_waiting_time + task.current_execute_time + task.transmission_waiting_time + task.current_transmission_time
            if task_time > task.task_deadline:
                task_total_cost += self.args.task_penalty
                drop_task_number += 1
                finished_task_number += 1
                task_completion_time += task_time
                task_execution_waiting_time += task.execute_waiting_time
                task_execution_time += task.current_execute_time
                task_transmission_waiting_time += task.transmission_waiting_time
                task_transmission_time += task.current_transmission_time
                task_reliability += task.reliability
            else:
                new_executed_queue.append(task)
            self.executed_queue = new_executed_queue
        return {
            "task_completion_time": task_completion_time,
            "task_total_cost": task_total_cost,
            "task_execution_waiting_time": task_execution_waiting_time,
            "task_execution_time": task_execution_time,
            "task_transmission_waiting_time": task_transmission_waiting_time,
            "task_transmission_time": task_transmission_time,
            "drop_task_number": drop_task_number,
            "finished_task_number": finished_task_number,
            "task_reliability": task_reliability,
        }

    def receive_task(self):
        #  对remote queue 的处理  物理上有N-1个队列  OFDMA
        for received_queue in self.received_queues:
            if len(received_queue) > 0:
                task = received_queue[0]
                task.current_transmission_time += self.args.mini_time_slot
                for i in range(1, len(received_queue)):
                    waiting_task = received_queue[i]
                    waiting_task.transmission_waiting_time += self.args.mini_time_slot
                if task.current_transmission_time >= task.transmission_time:
                    self.executed_queue.append(received_queue.popleft())

    def send_task(self):
        # 对 send queue 的处理 物理上有N-1个队列  OFDMA
        for send_queue in self.sent_queues:
            if len(send_queue) > 0:
                task = send_queue[0]
                task.current_transmission_time += self.args.mini_time_slot
                for i in range(1, len(send_queue)):
                    waiting_task = send_queue[i]
                    waiting_task.transmission_waiting_time += self.args.mini_time_slot
                if task.current_transmission_time >= task.transmission_time:
                    send_queue.popleft()
