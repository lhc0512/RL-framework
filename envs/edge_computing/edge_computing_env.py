import numpy as np

from envs.edge_computing.edge_node import EdgeNode
from envs.edge_computing.multiagentenv import MultiAgentEnv
import math


class EdgeComputingEnv(MultiAgentEnv):
    def __init__(self, args):
        self.edge_nodes = []
        for i in range(args.edge_node_num):
            # pass the parameters.
            edge_node = EdgeNode(args)
            # set the edge node id.
            edge_node.id = i
            self.edge_nodes.append(edge_node)
        # the distance between edge nodes.
        d = [[0 for _ in range(args.edge_node_num)] for _ in range(args.edge_node_num)]
        # the transmission rate between edge nodes.
        R = [[0 for _ in range(args.edge_node_num)] for _ in range(args.edge_node_num)]
        for i in range(args.edge_node_num):
            for j in range(args.edge_node_num):
                # There may be zero, so we add one.
                d[i][j] = ((self.edge_nodes[i].x - self.edge_nodes[j].x) ** 2 + (self.edge_nodes[i].y - self.edge_nodes[j].y) ** 2) ** 0.5 + 1
                # channel gain
                channel_gain = 1 / d[i][j] ** 4
                # transmission rate
                R[i][j] = args.B * 1024 * 1024 * math.log2(1 + args.P * channel_gain / args.P_n)
        for i in range(args.edge_node_num):
            self.edge_nodes[i].init_edge_node_transmission_rates(R[i])

        self.current_step = 0
        for i in range(args.edge_node_num):
            self.edge_nodes[i].generate_task()
        # 方便节点之间的通信编写
        for i in range(args.edge_node_num):
            self.edge_nodes[i].edge_nodes = self.edge_nodes

        # 没有用户任务，不做卸载决策，占最后一位
        args.n_actions = args.edge_node_num + 1
        args.n_agents = args.edge_node_num
        self.args = args

        #  设置obs 和 state 的大小
        self.obs_shape = self.edge_nodes[0].obs_shape
        self.state_shape = self.obs_shape * self.args.edge_node_num

    def get_obs(self):
        """Returns all agent observations in a list"""
        agent_obs_list = []
        for i in range(self.args.edge_node_num):
            agent_obs_list.append(self.edge_nodes[i].get_observation())
        return np.array(agent_obs_list)

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        obs = self.edge_nodes[agent_id].get_observation()
        raise np.array(obs)

    def get_obs_size(self):
        """Returns the shape of the observation"""
        raise self.obs_shape

    def get_state(self):
        state = []
        for i in range(self.args.edge_node_num):
            state.extend(self.edge_nodes[i].get_observation())
        return np.array(state)

    def get_state_size(self):
        """Returns the shape of the state"""
        raise self.state_shape

    def get_avail_agent_actions(self, agent_id):
        return self.edge_nodes[agent_id].get_avail_actions()

    def get_avail_actions(self):
        result = []
        for i in range(self.args.edge_node_num):
            result.append(self.get_avail_agent_actions(i))
        return result

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.args.edge_node_num + 1

    def step(self, actions):
        step_finished_task_number = 0
        step_drop_task_number = 0
        step_total_task_completion_time = 0
        step_total_task_reliability = 0
        step_total_task_total_cost = 0
        # TODO debug
        step_total_task_transmission_waiting_time = 0
        step_total_task_transmission_time = 0
        step_total_task_execution_waiting_time = 0
        step_total_task_execution_time = 0
        for i in range(len(actions)):
            self.edge_nodes[i].offload_task(actions[i])

        # let the simulation more closed to real environment
        for i in range(self.args.mini_time_slot_num):
            for edge_node in self.edge_nodes:
                info = edge_node.execute_task()
                step_finished_task_number += info["finished_task_number"]
                step_drop_task_number += info["drop_task_number"]
                step_total_task_completion_time += info["task_completion_time"]
                step_total_task_reliability += info["task_reliability"]
                step_total_task_total_cost += info["task_total_cost"]
                step_total_task_transmission_waiting_time += info["task_transmission_waiting_time"]
                step_total_task_transmission_time += info["task_transmission_time"]
                step_total_task_execution_waiting_time += info["task_execution_waiting_time"]
                step_total_task_execution_time += info["task_execution_time"]

            for edge_node in self.edge_nodes:
                edge_node.receive_task()

            for edge_node in self.edge_nodes:
                edge_node.send_task()

        self.current_step += 1
        for edge_node in self.edge_nodes:
            edge_node.generate_task()

        info = {
            "step_total_task_total_cost": step_total_task_total_cost,
            "step_finished_task_number": step_finished_task_number,
            "step_drop_task_number": step_drop_task_number,
            "step_total_task_completion_time": step_total_task_completion_time,
            "step_total_task_reliability": step_total_task_reliability,
            # TODO debug
            "step_total_task_execution_waiting_time": step_total_task_execution_waiting_time,
            "step_total_task_execution_time": step_total_task_execution_time,
            "step_total_task_transmission_waiting_time": step_total_task_transmission_waiting_time,
            "step_total_task_transmission_time": step_total_task_transmission_time,
        }

        team_reward = -step_total_task_total_cost

        if self.current_step >= self.args.episode_limit:
            info["episode_limit"] = True
            terminated = True
            # # 防止所有节点将task offloading到同一个节点中, 因为没有完成任务reward=0, 完成任务reward<0
            # for edge_node in self.edge_nodes:
            #     for task in edge_node.executed_queue:
            #         task_time = task.current_transmission_time + task.transmission_waiting_time + task.execute_waiting_time + task.current_execute_time
            #         if task_time > task.task_deadline:
            #             team_reward += - self.args.task_penalty
        else:
            info["episode_limit"] = False
            terminated = False
        return team_reward, terminated, info

    def reset(self):
        self.current_step = 0
        for edge_node in self.edge_nodes:
            edge_node.reset_edge_node()

    def close(self):
        pass

    def get_env_info(self):
        return {
            "state_shape": self.state_shape,
            "obs_shape": self.obs_shape,
            "n_actions": self.args.edge_node_num + 1,
            "n_agents": self.args.edge_node_num,
            "episode_limit": self.args.episode_limit,
        }

    def save_replay(self):
        pass

    def get_stats(self):
        return {
            "message": "test_messages",
        }
