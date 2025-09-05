import math
import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from .model import QNetwork
from .replay_buffer import ReplayBuffer


BUFFER_SIZE = 10000         # replay buffer size
BATCH_SIZE = 32            # minibatch size
GAMMA = 0.995 #1.00              # discount factor
TAU = 5e-3                # for soft update of target parameters
LR = 1e-4                 # learning rate
CLIP_GRAD_NORM_VALUE = 5  # value of gradient to clip while training
UPDATE_TARGET_EACH = 500  # number of steps to wait until updating target network
UPDATE_PARAMS_EACH = 4    # number of steps to wait until sampling experience tuples and updating model params
WARMUP_STEPS = 1000       # number of steps to wait before start learning

USE_NEW_EDGE_Q_LAYER = True  # if False, use the old QLayer that does not consider edge features

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        n_node_features,
        n_edge_features,
        nstep=1,
        embedding_dim=256,
        embedding_layers=4,
        normalize=True,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        lr=LR,
        clip_grad_norm_value=CLIP_GRAD_NORM_VALUE,
        update_target_each=UPDATE_TARGET_EACH,
        target_update="soft",
        update_params_each=UPDATE_PARAMS_EACH,
        warmup_steps=WARMUP_STEPS,
        double_dqn=False,
    ):
        """Initialize an Agent object"""

        self.nstep = nstep
        self.use_nstep = nstep > 1
        self.double_dqn = double_dqn

        self.gamma = gamma
        self.clip_grad_norm_value = clip_grad_norm_value
        self.update_target_each = update_target_each
        self.update_params_each = update_params_each
        self.warmup_steps = warmup_steps
        self.tau = tau
        self.target_update = target_update
        assert target_update in ("soft", "hard"), 'target_update must be one of {"soft", "hard"}'

        # Q-Network
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.qnetwork_local = QNetwork(embed_dim=embedding_dim, embedding_layers=embedding_layers, n_node_features=n_node_features,
                                       n_edge_features=n_edge_features, normalize=normalize, use_new_edge_q_layer=USE_NEW_EDGE_Q_LAYER).to(device, dtype=torch.float32)
        self.qnetwork_target = QNetwork(embed_dim=embedding_dim, embedding_layers=embedding_layers, n_node_features=n_node_features,
                                        n_edge_features=n_edge_features, normalize=normalize, use_new_edge_q_layer=USE_NEW_EDGE_Q_LAYER).to(device, dtype=torch.float32)

        self.hard_update(self.qnetwork_local, self.qnetwork_target)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.global_t_step = 0
        self.update_t_step = 0

        #if USE_NEW_EDGE_Q_LAYER:
        #    self.edges_ij = torch.as_tensor(self.valid_edges, dtype=torch.long, device=device)  # (E,2)


        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size)

        # To be used in n-step learning
        self.gamma_n_minus_1 = self.gamma ** (self.nstep - 1)
        self.gamma_n = self.gamma ** self.nstep

        # Internal values accross steps
        self.episode_losses = []
        self.losses = []
        self.q_expecteds = []
        self.q_targets = []
        self.params = []
        # self.grads = []
        self.theta1s = []
        self.theta2s = []
        self.theta3s = []
        self.theta4s = []
        self.theta5s = []
        self.theta6s = []
        self.theta7s = []
        # Initial episode config
        self.reset_episode()

    def reset_episode(self):
        self.espisode_t_step = 0
        self.states = deque(maxlen=self.nstep)
        self.edge_features = deque(maxlen=self.nstep)
        self.actions = deque(maxlen=self.nstep)
        self.rewards = deque(maxlen=self.nstep)
        self.sum_rewards = 0

        if self.qnetwork_local.training:
            self.losses.append(sum(self.episode_losses) / len(self.episode_losses) if len(self.episode_losses) > 0 else 0)
            self.episode_losses = []

    def set_valid_edges(self, valid_edges):
        """
        valid_edges: np.ndarray or list of shape (E, 2) with node indices for each controllable edge
        """
        self.valid_edges = np.asarray(valid_edges, dtype=np.int64)
        self.E = self.valid_edges.shape[0]
        self.no_op_index = self.E
        # Cache edges_ij on the right device
        self.edges_ij = torch.as_tensor(self.valid_edges, dtype=torch.long, device=device)

    @torch.no_grad()
    def act(self, state, edge_feature, eps=0.0):
        """
        Returns an int in [0..E], where E is the no-op.
        """
        state_t = torch.from_numpy(state).unsqueeze(0).float().to(device)
        ef_t    = torch.from_numpy(edge_feature).unsqueeze(0).float().to(device)

        # Q over valid edges + no-op: (E+1,)
        q = self.qnetwork_local(state_t, ef_t, self.edges_ij).squeeze(0)

        if random.random() < eps:
            # simple epsilon-greedy over E+1 actions
            return int(np.random.randint(self.E + 1))
        else:
            return int(torch.argmax(q).item())

    def step(self, state, edge_feature, action, reward, next_state, next_edge_feature, done):
        self.espisode_t_step += 1
        self.global_t_step += 1
        if not self.use_nstep:
            # Save experience in replay memory
            self.memory.add(state, edge_feature, action, reward, next_state, next_edge_feature, done)
            if len(self.memory) >= self.memory.batch_size \
                    and self.global_t_step % self.update_params_each == 0 \
                    and self.global_t_step >= self.warmup_steps:
                experiences = self.memory.sample()
                self.learn(experiences)
        else:
            reward_to_subtract = self.rewards[0] if self.espisode_t_step > self.nstep else 0  # r1

            self.states.append(state)
            self.edge_features.append(edge_feature)
            self.actions.append(action)
            self.rewards.append(reward)

            if len(self.rewards) <= self.nstep:
                self.sum_rewards += reward * (self.gamma ** (len(self.rewards)-1))
            else:
                # reward_to_subtract = self.rewards[0]
                self.sum_rewards = (self.sum_rewards - reward_to_subtract) / self.gamma
                self.sum_rewards += reward * self.gamma_n_minus_1

            # Get xv from info
            if self.espisode_t_step >= self.nstep:
                # Get oldest state and action (S_{t-n}, a_{t-n}) to add to replay memory buffer
                oldest_state = self.states[0]
                oldest_edge_feature = self.edge_features[0]
                oldest_action = self.actions[0]

                # Save experience in replay memory
                self.memory.add(
                    oldest_state,
                    oldest_edge_feature,
                    oldest_action,
                    self.sum_rewards,
                    next_state,
                    next_edge_feature,
                    done
                )

            # Different from the paper, as it should be called inside (if self.t_step >= self.nstep)
            if len(self.memory) >= self.memory.batch_size \
                    and self.global_t_step % self.update_params_each == 0 \
                    and self.global_t_step >= self.warmup_steps:
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        states, edge_features, actions, rewards, next_states, next_edge_features, dones = experiences

        actions   = actions.view(-1)                           # (B,)
        B         = actions.size(0)
        batch_idx = torch.arange(B, device=actions.device)

        # ---- Q(s,a) ----
        q_pred    = self.qnetwork_local(states, edge_features, self.edges_ij)   # (B, E+1)
        q_expected = q_pred[batch_idx, actions].unsqueeze(1)                    # (B, 1)

        # ---- Targets ----
        with torch.no_grad():
            target_q = self.qnetwork_target(next_states, next_edge_features, self.edges_ij)  # (B, E+1)

            if self.double_dqn:
                local_q  = self.qnetwork_local(next_states, next_edge_features, self.edges_ij)  # (B, E+1)
                a_star   = local_q.argmax(dim=1)                                                # (B,)
                q_t_next = target_q[batch_idx, a_star].unsqueeze(1)                             # (B,1)
            else:
                q_t_next = target_q.max(dim=1, keepdim=True)[0]                                 # (B,1)

        q_targets = rewards + (self.gamma_n * q_t_next * (1 - dones))                           # (B,1)

        # ---- Loss & update ----
        loss = F.huber_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        self.episode_losses.append(loss.item())
        loss.backward()
        if self.clip_grad_norm_value is not None:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.clip_grad_norm_value)
        self.optimizer.step()

        # ---- Target network update ----
        if self.target_update == "soft":
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        else:
            self.update_t_step = (self.update_t_step + 1) % self.update_target_each
            if self.update_t_step == 0:
                self.hard_update(self.qnetwork_local, self.qnetwork_target)


    def _log_params(self, actions, dones, loss, next_states, q_expected, q_targets, states, target_preds):
        if loss.item() > 5e150:
            print(f'actions: {list(actions.cpu().detach().numpy().flatten())}')
            print(f'q_expected: {list(q_expected.cpu().detach().numpy().flatten())}')
            print(f'q_targets: {list(q_targets.cpu().detach().numpy().flatten())}')
            print(f'{loss=}')
            print(f'{target_preds=}')
            print(f'{dones=}')
            print(f'{1 - dones=}')
            print(f'{states=}')
            print(f'{next_states=}')
            print(f'{self.espisode_t_step=}')
            sys.exit(0)
        self.q_targets.append(q_targets.min().item())
        self.q_expecteds.append(q_expected.min().item())
        # self.params.append(next(self.qnetwork_local.parameters())[0,0].item())
        self.theta1s.append(self.qnetwork_local.node_features_embedding_layer.theta1.weight[0, 0].item())
        self.theta2s.append(self.qnetwork_local.embedding_layer.theta2.weight[0, 0].item())
        self.theta3s.append(self.qnetwork_local.edge_features_embedding_layer.theta3.weight[0, 0].item())
        # self.theta4s.append(self.qnetwork_local.edge_features_embedding_layer.theta4.weight[0,0].item())
        self.theta5s.append(self.qnetwork_local.q_layer.theta5.weight[0, 0].item())
        self.theta6s.append(self.qnetwork_local.q_layer.theta6.weight[0, 0].item())
        self.theta7s.append(self.qnetwork_local.q_layer.theta7.weight[0, 0].item())

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def hard_update(local_model, target_model):
        """Hard update model parameters.
        θ_target = θ_local

        Inputs:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        target_model.load_state_dict(local_model.state_dict())

    def train(self):
        """Configure PyTorch modules to be in train mode"""
        self.qnetwork_target.train()
        self.qnetwork_local.train()

    def eval(self):
        """Configure PyTorch modules to be in eval mode"""
        self.qnetwork_target.eval()
        self.qnetwork_local.eval()

    def load_model(self, checkpoint_path):
        """Load model's checkpoint"""
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.global_t_step = checkpoint['global_t_step']
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_model(self, checkpoint_path):
        """Save model's checkpoint"""
        print(f"Saving model to {checkpoint_path}")
        checkpoint = {
            'global_t_step': self.global_t_step,
            'qnetwork_local': self.qnetwork_local.state_dict(),
            'qnetwork_target': self.qnetwork_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)





'''
    @torch.no_grad()
    def old_act(self, state, edge_feature, *args, **kwargs):
        """Returns actions for given state as per current policy.

        Params
        ======
            obs        : current observation
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state_t = torch.from_numpy(state).unsqueeze(0).float().to(device)
        ef_t    = torch.from_numpy(edge_feature).unsqueeze(0).float().to(device)

        if USE_NEW_EDGE_Q_LAYER:
            qmat = self.qnetwork_local(state_t, ef_t, self.edges_ij).squeeze(0)  # (E+1,)
        else:
            qmat = self.qnetwork_local(state_t, ef_t).squeeze(0)  # torch.Tensor (N*N+1)
        
        M     = qmat.shape[0]  # M = N*N + 1
        E     = len(self.valid_edges)
        N = int(math.sqrt(M - 1))  # N*N+1 = M, so N = sqrt(M-1)
        no_op_idx = M - 1  # no-op action is the last one

        idx = None
        #print(f"M={M}, E={E}, qmat.shape={qmat.shape}")

        eps = kwargs.get("eps", 0.0)
        if random.random() < eps:
            p_noop = 0.3
            if random.random() < p_noop:
                print(f"Random No-op action selected - {E}")
                idx = E
            else: 
                idx = np.random.randint(E)
                print(f"Random action selected ={idx}")
        else:
            edge_qs = qmat[self.valid_edges_linear]  # shape [E]
            best_edge_pos = int(edge_qs.argmax())   # 0..E-1
            if qmat[ no_op_idx ] > edge_qs[best_edge_pos]:
                print(f"## Agent No-op action selected - {no_op_idx}")
                idx = E  #the no-op index
            else:
                idx = best_edge_pos  
                print(f"## Agent action selected - {idx} (edge {self.valid_edges[idx]})")

        return int(idx)

        
        #state = torch.from_numpy(state).to(device, dtype=torch.float32)
        #edge_feature = torch.from_numpy(edge_feature).to(device, dtype=torch.float32)
        ## adjacency lives in your `state` matrix after the first n_node_features columns
        #
        #valid_edges = list(self.edge_map.keys()) 
        #eps = kwargs.get("eps", 0.0)
        #if random.random() < eps:
        #    idx = np.random.randint(len(valid_edges))
        #    return valid_edges[idx].tolist()  # returns [i, j]
        #action_values = self.qnetwork_local(state, edge_feature).squeeze(0)  # (N, N)
        #scores = action_values[valid_edges[:, 0], valid_edges[:, 1]]
        #best_idx = scores.argmax()
        #best_action = valid_edges[best_idx].tolist()
        #return best_action

    @torch.no_grad()
    def old_act(self, state, edge_feature, *args, **kwargs):
        """Returns actions for given state as per current policy.

        Params
        ======
            obs        : current observation
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).to(device, dtype=torch.float32)
        edge_feature = torch.from_numpy(edge_feature).to(device, dtype=torch.float32)

        # Valid actions are nodes that aren´t already in the partial solution
        #xv = state[:, 0]
        #valid_actions = (xv == 0).nonzero()
        #action_values = self.qnetwork_local(state, edge_feature).squeeze(0)  # squeeze to remove NN batching
        #valid_actions_idx = action_values[valid_actions].argmax().item()
        #action = valid_actions[valid_actions_idx].item()
        #return action

        edge_state = edge_feature[0, :, :, 0]  # e.g., 1=open, 0=closed
        valid_actions = edge_state.nonzero(as_tuple=False)


        # Epsilon-greedy: greedy action selection
        eps = kwargs.get("eps", 0.0)
        if random.random() < eps:
            action_idx = np.random.randint(len(valid_actions))
            action = valid_actions[action_idx].item()
            return action

        action_values = self.qnetwork_local(state, edge_feature).squeeze(0)  # shape: (N, N)
        valid_scores = action_values[valid_actions[:, 0], valid_actions[:, 1]]
        best_idx = valid_scores.argmax()
        best_action = valid_actions[best_idx].tolist()  # returns [i, j]
        return best_action

    def old_old_learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) 
        """
        states, edge_features, actions, rewards, next_states, next_edge_features, dones = experiences
        batch_size = actions.size(0)

        # -------------------- Q(s,a) --------------------
        q_pred = self.qnetwork_local(states, edge_features)  # (B, N*N +1)
        #i, j = actions[:, 0], actions[:, 1]
        #batch_indices = torch.arange(batch_size, device=actions.device)
        batch_indices = torch.arange(batch_size, device=actions.device)
        q_expected = q_pred[batch_indices, actions.squeeze(-1)].unsqueeze(1)  # (B, 1)

        # -------------------- Q-target --------------------
        with torch.no_grad():
            if self.double_dqn:
                # a_max = argmax_a Q_local(s', a)
                local_q = self.qnetwork_local(next_states, next_edge_features)  # (B, N, N)
                a_max = local_q.view(batch_size, -1).argmax(dim=1)  # index in flattened N x N
                a_max_i = a_max // local_q.shape[2]
                a_max_j = a_max % local_q.shape[2]

                # Q_target(s', a_max)
                target_q = self.qnetwork_target(next_states, next_edge_features)  # (B, N, N)
                q_targets_next = target_q[batch_indices, a_max_i, a_max_j].unsqueeze(1)  # (B, 1)
            else:
                target_q = self.qnetwork_target(next_states, next_edge_features)  # (B, N, N)
                q_targets_next = target_q.view(batch_size, -1).max(dim=1, keepdim=True)[0]  # (B, 1)

        # TD target
        q_targets = rewards + self.gamma_n * q_targets_next * (1 - dones)

        # -------------------- Loss --------------------
        loss = F.huber_loss(q_expected, q_targets)

        self._log_params(actions, dones, loss, next_states, q_expected, q_targets, states, target_q)

        self.optimizer.zero_grad()
        self.episode_losses.append(loss.item())
        loss.backward()
        if self.clip_grad_norm_value is not None:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.clip_grad_norm_value)
        self.optimizer.step()

        # -------------------- Target network update --------------------
        if self.target_update == "soft":
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        elif self.target_update == "hard":
            self.update_t_step = (self.update_t_step + 1) % self.update_target_each
            if self.update_t_step == 0:
                self.hard_update(self.qnetwork_local, self.qnetwork_target)

    def old_learn(self, experiences):
        states, edge_features, actions, rewards, next_states, next_edge_features, dones = experiences
        actions = actions.view(-1)              # (B,)
        B = actions.size(0)
        batch_idx = torch.arange(B, device=actions.device)

        # Forward
        q_pred = self.qnetwork_local(states, edge_features)   # (B, M=N*N+1)
        M = q_pred.size(1)
        no_op_idx = M - 1

        # Map replay "edge-index or no-op" -> absolute column in q_pred
        # self.valid_edges_linear is set by env (shape [E]), keep it on the same device:
        edge_lin = self.valid_edges_linear.to(actions.device)  # (E,)
        E = edge_lin.size(0)

        is_noop = (actions == E)
        edge_pos = torch.clamp(actions, max=E-1)              # for non-noop rows
        a_abs = torch.where(is_noop, torch.full_like(actions, no_op_idx), edge_lin[edge_pos])

        q_expected = q_pred[batch_idx, a_abs].unsqueeze(1)    # (B,1)

        # ---- Targets: mask to valid edges + no-op ----
        with torch.no_grad():
            target_q_all = self.qnetwork_target(next_states, next_edge_features)  # (B,M)

            # Gather only valid columns
            target_edge = target_q_all[:, edge_lin]                                # (B,E)
            target_noop = target_q_all[:, no_op_idx].unsqueeze(1)                  # (B,1)
            target_masked = torch.cat([target_edge, target_noop], dim=1)           # (B,E+1)

            if self.double_dqn:
                local_q_all = self.qnetwork_local(next_states, next_edge_features) # (B,M)
                local_edge = local_q_all[:, edge_lin]                               # (B,E)
                local_noop = local_q_all[:, no_op_idx].unsqueeze(1)                 # (B,1)
                local_masked = torch.cat([local_edge, local_noop], dim=1)           # (B,E+1)

                a_rel = local_masked.argmax(dim=1)                                  # 0..E (E == no-op)
                a_abs_next = torch.where(a_rel == E, torch.full((B,), no_op_idx, device=actions.device), edge_lin[a_rel])
                q_targets_next = target_q_all[batch_idx, a_abs_next].unsqueeze(1)
            else:
                q_targets_next = target_masked.max(dim=1, keepdim=True)[0]

        q_targets = rewards + (self.gamma_n * q_targets_next * (1 - dones))

        loss = F.huber_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        self.episode_losses.append(loss.item())
        loss.backward()
        if self.clip_grad_norm_value is not None:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.clip_grad_norm_value)
        self.optimizer.step()

        # target update
        if self.target_update == "soft":
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        else:
            self.update_t_step = (self.update_t_step + 1) % self.update_target_each
            if self.update_t_step == 0:
                self.hard_update(self.qnetwork_local, self.qnetwork_target)

    def old_old_learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) 
        """
        states, edge_features, actions, rewards, next_states, next_edge_features, dones = experiences

        actions = actions.view(-1)  # shape: (batch_size,)
        batch_size = actions.size(0)
        batch_idx = torch.arange(batch_size, device=actions.device)

        # -------------------- Q(s,a) --------------------
        # Q-network outputs shape (batch, M) where M = N*N + 1
        q_pred = self.qnetwork_local(states, edge_features)  # (B, M)
        # Gather the Q-values corresponding to the taken actions
        q_expected = q_pred[batch_idx, actions].unsqueeze(1)  # (B, 1)

        # -------------------- Q-target --------------------
        with torch.no_grad():
            target_q = self.qnetwork_target(next_states, next_edge_features)  # (B, M)
            if self.double_dqn:
                # Double DQN: action selection by local network, evaluation by target network
                local_q_next = self.qnetwork_local(next_states, next_edge_features)  # (B, M)
                a_max = local_q_next.argmax(dim=1)  # (B,)
                q_targets_next = target_q[batch_idx, a_max].unsqueeze(1)  # (B, 1)
            else:
                # Standard DQN: max over next-state Q-values
                q_targets_next = target_q.max(dim=1, keepdim=True)[0]  # (B, 1)

        # Compute TD target
        q_targets = rewards + (self.gamma_n * q_targets_next * (1 - dones))  # (B, 1)

        # -------------------- Loss --------------------
        loss = F.huber_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        self.episode_losses.append(loss.item())
        loss.backward()

        # Gradient clipping
        if self.clip_grad_norm_value is not None:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.clip_grad_norm_value)
        self.optimizer.step()

        # -------------------- Target network update --------------------
        if self.target_update == "soft":
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        else:
            self.update_t_step = (self.update_t_step + 1) % self.update_target_each
            if self.update_t_step == 0:
                self.hard_update(self.qnetwork_local, self.qnetwork_target)
'''