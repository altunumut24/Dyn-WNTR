from collections import OrderedDict, namedtuple
import copy
from email.mime import base
import random
import sys
import math
import numpy as np
import torch

sys.path.append("../..")
#from modules.agents.dqn_agent import DQNAgent
#from wntr.sim.interactive_network_simulator import InteractiveWNTRSimulator
#from wntr.network.model import WaterNetworkModel
#from wntr.network.elements import LinkStatus, Junction

from modules.rl.agents.dqn_agent import DQNAgent
from mwntr.sim.interactive_network_simulator import InteractiveWNTRSimulator
from mwntr.network.model import WaterNetworkModel
from mwntr.network.elements import LinkStatus, Junction


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

EnvInfo = namedtuple("EnvInfo", field_names=["observation", "reward", "done"])

class WDNEnv():
    agent: DQNAgent
    simulation: InteractiveWNTRSimulator
    base_wn: WaterNetworkModel
    n_node_features: int
    n_edge_features: int
    normalize_reward: bool = True
    node_list: list
    node_idx: dict
    edge_list: list
    edge_map: dict
    global_timestep: int
    duration: int

    def __init__(self, base_wn=None, normalize_reward=True, double_dqn=False, global_timestep=60, simulation_duration=86400):
        """
        Initialize the WDN environment with the given parameters.
        :param n_node_features: Number of node features
        :param n_edge_features: Number of edge features
        :param graph: NetworkX graph representing the water distribution network
        :param normalize_reward: Whether to normalize the reward
        """

        

        #super().__init__()
        self.simulation = None
        self.normalize_reward = normalize_reward
        self.double_dqn = double_dqn

        wn = None
        if base_wn is not None:
            self.base_wn = copy.deepcopy(base_wn)
            wn = self.base_wn
        else:
            raise ValueError("Base WaterNetworkModel must be provided")

        # list of all junction/reservoir/tank names in order
        self.node_list = list(wn.node_name_list)
        self.node_idx  = {name:i for i,name in enumerate(self.node_list)}
        # list of all links (pipes, valves, pumps…)
        self.edge_list = list(wn.link_name_list)
        # map (u_idx,v_idx) → link_name (and vice-versa)
        self.edge_map = {}
        for link_name in self.edge_list:
            link = wn.get_link(link_name)
            u = self.node_idx[link.start_node_name]
            v = self.node_idx[link.end_node_name]
            self.edge_map[(u,v)] = link_name
            self.edge_map[(v,u)] = link_name
        self.global_timestep = global_timestep
        self.duration = simulation_duration
        self.leak_nodes = []
        self.demanding_nodes = []
        self.closed_links = 0

        self.node_features = ['demand', 'diameter', 'elevation', 'has_diameter', 'has_elevation', 'has_level', 'has_max_level', 'has_min_level', 'has_overflow', 'has_setting', 'head', 'leak_area', 'leak_demand', 'leak_discharge_coeff', 'leak_status', 'level', 'max_level', 'min_level', 'node_type', 'overflow', 'pressure', 'setting']
        self.n_node_features = len(self.node_features) + 2 #one-hot for element_type (junction, reservoir, tank)
        self.edge_features = ['base_speed', 'diameter', 'flow', 'has_base_speed', 'has_diameter', 'has_headloss', 'has_roughness', 'has_setting', 'has_velocity', 'headloss', 'link_type', 'roughness', 'setting', 'status', 'velocity']
        self.n_edge_features = len(self.edge_features) + 2 #one-hot for element_type (pipe, valve, pump)

        self.agent = DQNAgent(n_node_features=self.n_node_features, n_edge_features=self.n_edge_features, nstep=10, double_dqn=self.double_dqn, embedding_dim=64)

        unique_edges = sorted({(min(u, v), max(u, v)) for (u, v) in self.edge_map.keys()})
        self.agent.set_valid_edges(np.array(unique_edges, dtype=np.int64))
        self.no_op_action = len(unique_edges)  # E -> no-op action is the last one

        #ve = torch.tensor(self.agent.valid_edges, device=device).long()  # shape [E,2]
        #self.agent.valid_edges_linear = ve[:,0]*len(self.node_list) + ve[:,1]  # shape [E]

        self.wn = wn

        self.episode_count = 0
        self.step_count = 0
        self.episode_log_file = None

        self.reset(global_timestep=global_timestep, duration=simulation_duration)

    def get_observation(self) -> tuple:
        try:
            snap = self.simulation.extract_snapshot()
        except Exception as e:
            print(f"#### Failed to extract snapshot: {e}")
            raise e
        
        # --- node features matrix ---
        node_feats = []
        for name, feats in snap['nodes'].items():
            i = self.node_idx[name]
            flat = []
            for f in self.node_features:
                v = feats[f]
                if isinstance(v, (list, tuple, np.ndarray)):
                    # a one‐hot vector: extend with all its entries
                    flat.extend(v)
                else:
                    # a single scalar
                    flat.append(v)
            node_feats.append(flat)

        state = np.asarray(node_feats, dtype=np.float32)

        # --- edge features tensor ---
        edge_feats = []
        for (i, j) in self.agent.valid_edges:
            link_name = self.edge_map.get((i, j), self.edge_map.get((j, i)))
            flat = []
            for f in self.edge_features:
                val = snap['edges'][link_name][f]
                if isinstance(val, (list, tuple, np.ndarray)):
                    flat.extend(val)
                else:
                    flat.append(val)
            edge_feats.append(flat)

        edge_feats = np.asarray(edge_feats, dtype=np.float32)
        assert edge_feats.shape[0] == len(self.agent.valid_edges), f"edge_feats has {edge_feats.shape[0]} rows, valid_edges has {len(self.agent.valid_edges)}"

        return state, edge_feats

    def reset(self, global_timestep=60, duration=86400) -> tuple:
        # initialize/reset the hydraulic sim

        #assert self.simulation_file_path is not None, "Simulation file path must be provided"
        wn = copy.deepcopy(self.base_wn)

        wn.add_pattern('constant', [1.0])  # add a constant pattern for demand
        wn.add_pattern('gaussian', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0])  # example pattern

        self.simulation = InteractiveWNTRSimulator(wn)
        self.simulation.init_simulation(global_timestep=global_timestep, duration=duration)

        self.demanding_nodes = []
        self.leak_nodes = []
        self.closed_links = 0

        self.episode_count += 1
        self.step_count = 0

        if self.episode_log_file:
            self.episode_log_file.close()
        #self.episode_log_file = open(f"single_round_logs/episode_{self.episode_count}.log", "w")
        self.episode_log_file = None

        #add 3 leaks and 5 demands to random nodes
        for _ in range(3):
            node_name = random.choice(list(filter(lambda x: x not in self.leak_nodes, self.wn.junction_name_list)))
            self.simulation.start_leak(node_name, leak_area=0.03)
            print(f"@@@@ Added leak at {node_name} @@@", file=self.episode_log_file if self.episode_log_file else sys.stdout)
            self.leak_nodes.append(node_name)
        
        for _ in range(5):
            node_name = random.choice(list(filter(lambda x: x not in self.leak_nodes and x not in self.demanding_nodes, self.wn.junction_name_list)))
            random_demand = 0.5 #random.uniform(0.1, 0.5)
            self.simulation.add_demand(node_name, base_demand=random_demand, name='constant', category='user_added')
            print(f"@@@@ Added demand at {node_name} @@@", file=self.episode_log_file if self.episode_log_file else sys.stdout)
            self.demanding_nodes.append(node_name)

        # return initial (state, edge_features)
        self.wn = wn

        self.simulation.step_sim()
        return self.get_observation()

    def calculate_reward(self, action) -> float:
        """
        Calculate the reward based on the current snapshot of the simulation.
        :param snap: Snapshot of the simulation containing node and edge data
        :return: Calculated reward
        """
        results = self.simulation.get_results()
        snap = self.simulation.extract_snapshot()
        print("Calculating reward...")

        #with open(f"single_round_logs/snapshot.json", "w") as f:
        #    import json
        #    json.dump(snap, f, indent=4)


        total_leak = sum(n['leak_demand'] for n in snap['nodes'].values() if n['leak_status'] == 1)
        satisfaction_nodes = [n['satisfied_demand'] for n in snap['nodes'].values() if n['expected_demand'] > 0.0001]
        print(f"Satisfaction nodes: {satisfaction_nodes}", file=self.episode_log_file if self.episode_log_file else sys.stdout)
        if len(satisfaction_nodes) == 0:
            average_satisfaction = 1
        else:
            average_satisfaction: np.floating[copy.Any] = np.average(satisfaction_nodes)
        print(f"Total Leak: {total_leak}, Average Satisfaction: {average_satisfaction}", file=self.episode_log_file if self.episode_log_file else sys.stdout)
        if len(self.leak_nodes) > 0:
            for node_name in self.leak_nodes:
                node = snap['nodes'][node_name]
                print(f"  Leak at {node_name}: area={node['leak_area']}, demand={node['leak_demand']}, status={node['leak_status']}", file=self.episode_log_file if self.episode_log_file else sys.stdout)
                #if node['leak_status'] == 1:
                #    total_leak += node['leak_demand']
            leaking_nodes = [(name, val["leak_demand"]) for (name, val) in snap['nodes'].items() if val['leak_status'] == 1]
            print(f"Leaking Nodes ({len(leaking_nodes)}): {leaking_nodes}", file=self.episode_log_file if self.episode_log_file else sys.stdout)
        
        unsatisfied_nodes = [(name, val["satisfied_demand"], val["expected_demand"]) for (name, val) in snap['nodes'].items() if val['expected_demand'] > 0.0001 and val['satisfied_demand'] < 0.999]
        print(f"Unsatisfied Nodes ({len(unsatisfied_nodes)}): {unsatisfied_nodes if len(unsatisfied_nodes) < 20 else 'Too many'} ", file=self.episode_log_file if self.episode_log_file else sys.stdout)
        print(f"Demanding nodes ({len(self.demanding_nodes)}): {self.demanding_nodes if len(self.demanding_nodes) < 20 else 'Too many'} ", file=self.episode_log_file if self.episode_log_file else sys.stdout)
        for node_name in self.demanding_nodes:
            node = snap['nodes'][node_name]
            print(f"  Demand at {node_name}: expected={node['expected_demand']}, satisfied={node['satisfied_demand']}") #, #file=self.episode_log_file if self.episode_log_file else sys.stdout)

        action_penalty = 0
        if (total_leak > 0 or average_satisfaction < 0.95) and action == self.no_op_action:
            action_penalty += 0.5
        elif total_leak == 0 and average_satisfaction >= 0.95 and action != self.no_op_action:
            action_penalty += 0.5

        average_satisfaction = (2 * average_satisfaction) - 1  # [-1, 1]
        leak_penalty = float(np.tanh(total_leak)) * 2  # 0.8 tunes how quickly penalty grows
        reward = -leak_penalty + float(average_satisfaction) - float(action_penalty) 
        print(f"Reward ({reward}) = -({leak_penalty}) + {average_satisfaction} - {action_penalty}", file=self.episode_log_file if self.episode_log_file else sys.stdout)

        return reward

    def add_random_event(self):
        if random.random() < 0.5:
            if random.random() < 0.5:
                # add a leak with some random parameters
                if len(self.leak_nodes) < 5:
                    node_name = random.choice(list(filter(lambda n: n not in self.leak_nodes and hasattr(self.wn.get_node(n), 'demand_timeseries_list'), self.node_list)))
                    self.simulation.start_leak(node_name, leak_area=0.03)
                    print(f"@@@@ Added leak at {node_name} @@@", file=self.episode_log_file if self.episode_log_file else sys.stdout)
                    self.leak_nodes.append(node_name)
            else:
                # remove a leak
                if len(self.leak_nodes) > 0:
                    node_name = random.choice(self.leak_nodes)
                    self.simulation.stop_leak(node_name)
                    print(f"@@@@ Removed leak at {node_name} @@@", file=self.episode_log_file if self.episode_log_file else sys.stdout)
                    self.leak_nodes.remove(node_name)
        else:
            if random.random() < 0.5:
                # add a random demand to a random node
                node_name = random.choice(list(filter(lambda n: n not in self.demanding_nodes and hasattr(self.wn.get_node(n), 'demand_timeseries_list'), self.node_list)))
                self.simulation.add_demand(node_name, base_demand=0.1, name='gaussian')
                print(f"@@@@ Added demand at {node_name} @@@", file=self.episode_log_file if self.episode_log_file else sys.stdout)
                self.demanding_nodes.append(node_name)
            else:
                # remove a demand
                if len(self.demanding_nodes) > 0:
                    node_name = random.choice(self.demanding_nodes)
                    self.simulation.remove_demand(node_name, name='gaussian')
                    print(f"@@@@ Removed demand at {node_name} @@@", file=self.episode_log_file if self.episode_log_file else sys.stdout)
                    self.demanding_nodes.remove(node_name)

    def step(self, action: int) -> EnvInfo:
        print(f"Action taken: {action} -- no-op={self.no_op_action}", file=self.episode_log_file if self.episode_log_file else sys.stdout)

        if action == self.no_op_action:
            print("No-op action taken, advancing simulation.", file=self.episode_log_file if self.episode_log_file else sys.stdout)
            # no action taken, just return current state
        else:
            u, v = self.agent.valid_edges[action]
            link_name = self.edge_map[(u, v)]
            link = self.simulation._wn.get_link(link_name)
            print(f"Current status: {link_name}->{link.status} (Closed = {LinkStatus.Closed}, open = {LinkStatus.Open}) ", file=self.episode_log_file if self.episode_log_file else sys.stdout)
            is_open = link.status != LinkStatus.Closed # 0=closed, else=open
            print(f"Action: {link_name} {'open' if not is_open else 'close'}", file=self.episode_log_file if self.episode_log_file else sys.stdout)
            if is_open:
                self.simulation.close_pipe(link_name)
                self.closed_links += 1
            else:
                self.simulation.open_pipe(link_name)
                self.closed_links -= 1

        print("Stepping simulation...")
        #if random.random() < 0.01:
        #    self.add_random_event()
        try: 
            self.simulation.step_sim()  # advance one hydraulic timestep
        except Exception as e:
            print(f"#### Simulation step failed: {e}")
            print(f"Simulation step failed: {e}", file=self.episode_log_file if self.episode_log_file else sys.stdout)
            return EnvInfo(self.get_observation(), -10000, True)
        

        next_state, next_edge_feats = self.get_observation()
        

        reward = self.calculate_reward(action)
        done = self.simulation.is_terminated()
        print(f"Next Step: Reward: {reward}, Done: {done}", file=self.episode_log_file if self.episode_log_file else sys.stdout)
        return EnvInfo((next_state, next_edge_feats), reward, done)


'''


    def different_get_observation(self):
        """
        Returns:
            state:        (N, F_node)     node features (NO dense adjacency concatenation)
            edge_feats:   (E, F_edge)     features only for the E canonical undirected edges
        """
        # ---- Node features (keep what you already had, just don't concat adjacency) ----
        node_feats = []
        for name in self.node_list:
            n = self.wn.get_node(name)
            # Example: keep your exact feature list/order here
            node_feats.append([
                getattr(n, "demand", 0.0) or 0.0,
                getattr(n, "head", 0.0) or 0.0,
                getattr(n, "pressure", 0.0) or 0.0,
                float(getattr(n, "_leak_status", False)),
                getattr(n, "_leak_area", 0.0) or 0.0,
                getattr(n, "_leak_demand", 0.0) or 0.0,
                # add/remove to match your previous node F_node precisely
            ])
        state = np.asarray(node_feats, dtype=np.float32)

        # ---- Sparse edge features aligned to self.agent.valid_edges (E, F_edge) ----
        edge_rows = []
        for (i, j) in self.agent.valid_edges:
            # Map canonical (i,j) back to a physical link name (either direction exists in edge_map)
            link_name = self.edge_map.get((i, j), self.edge_map.get((j, i)))
            l = self.wn.get_link(link_name)
            # Compute velocity if diameter present
            diameter = (getattr(l, "diameter", None) or 0.0)
            flow = getattr(l, "flow", 0.0) or 0.0
            velocity = (abs(flow) * 4.0 / (math.pi * diameter ** 2)) if diameter else 0.0
            status = 0.0 if l.status == LinkStatus.Closed else 1.0

            # IMPORTANT: keep this list/ordering identical to what your model expects (n_edge_features)
            edge_rows.append([
                status,
                flow,
                getattr(l, "headloss", 0.0) or 0.0,
                getattr(l, "roughness", 0.0) or 0.0,
                diameter,
                velocity,
                # add/remove to match your previous F_edge exactly
            ])
        edge_feats = np.asarray(edge_rows, dtype=np.float32)

        return state, edge_feats

    def old_get_observation(self) -> tuple:
        snap = self.simulation.extract_snapshot()
        N = len(self.node_list)

        # --- node features matrix ---
        node_feats = np.zeros((N, self.n_node_features), dtype=float)
        for name, feats in snap['nodes'].items():
            i = self.node_idx[name]
            flat = []
            for f in self.node_features:
                v = feats[f]
                if isinstance(v, (list, tuple, np.ndarray)):
                    # a one‐hot vector: extend with all its entries
                    flat.extend(v)
                else:
                    # a single scalar
                    flat.append(v)
            node_feats[i] = np.array(flat, dtype=float)

        # --- adjacency rows (open pipes only) ---
        adj = np.zeros((N, N), dtype=float)
        for (u,v), link_name in self.edge_map.items():
            status = snap['edges'][link_name]['status']
            adj[u,v] = status  # 1=open, 0=closed

        # stack node_feats ∥ adjacency
        state = np.hstack([node_feats, adj])

        # --- edge features tensor ---
        edge_feats = np.zeros((N, N, self.n_edge_features), dtype=float)
        for (u,v), link_name in self.edge_map.items():
            flat = []
            for f in self.edge_features:
                val = snap['edges'][link_name][f]
                if isinstance(val, (list, tuple, np.ndarray)):
                    flat.extend(val)
                else:
                    flat.append(val)
            edge_feats[u, v] = np.array(flat, dtype=float)

        return state, edge_feats
'''
