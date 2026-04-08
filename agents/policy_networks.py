"""
Policy Networks for SFJSSP DRL Agents

Evidence Status:
- Graph neural networks for scheduling: CONFIRMED from recent DRL literature
- Actor-critic architecture: CONFIRMED (PPO, A2C)
- Application to SFJSSP: PROPOSED

Note: Requires PyTorch and optionally PyTorch Geometric
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Try to import torch, provide fallback if not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create stub classes for when torch is not available
    class nn:
        Module = object
        Linear = object
        LayerNorm = object
        Dropout = object
    class F:
        @staticmethod
        def relu(x): return x
        @staticmethod
        def softmax(x, dim=-1): return x


class SFJSSPGraphEncoder:
    """
    Graph neural network encoder for SFJSSP state representation

    Encodes heterogeneous graph with:
    - Job nodes
    - Operation nodes
    - Machine nodes
    - Worker nodes

    Evidence: Graph-based state representation confirmed from DRL scheduling literature

    Args:
        job_feature_dim: Dimension of job features
        op_feature_dim: Dimension of operation features
        machine_feature_dim: Dimension of machine features
        worker_feature_dim: Dimension of worker features
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
    """

    def __init__(
        self,
        job_feature_dim: int = 6,
        op_feature_dim: int = 6,
        machine_feature_dim: int = 4,
        worker_feature_dim: int = 4,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        if not TORCH_AVAILABLE:
            return

        self.job_feature_dim = job_feature_dim
        self.op_feature_dim = op_feature_dim
        self.machine_feature_dim = machine_feature_dim
        self.worker_feature_dim = worker_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Feature encoders for each node type
        self.job_encoder = nn.Sequential(
            nn.Linear(job_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.op_encoder = nn.Sequential(
            nn.Linear(op_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.machine_encoder = nn.Sequential(
            nn.Linear(machine_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.worker_encoder = nn.Sequential(
            nn.Linear(worker_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Cross-attention for message passing (simplified)
        self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=4, batch_first=True)

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode(
        self,
        job_features: torch.Tensor,
        op_features: torch.Tensor,
        machine_features: torch.Tensor,
        worker_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode heterogeneous graph into node embeddings

        Args:
            job_features: (batch, n_jobs, job_feature_dim)
            op_features: (batch, n_ops, op_feature_dim)
            machine_features: (batch, n_machines, machine_feature_dim)
            worker_features: (batch, n_workers, worker_feature_dim)

        Returns:
            Dict with encoded embeddings for each node type
        """
        # Encode each node type
        job_emb = self.job_encoder(job_features)
        op_emb = self.op_encoder(op_features)
        machine_emb = self.machine_encoder(machine_features)
        worker_emb = self.worker_encoder(worker_features)

        # Simple message passing via attention (simplified)
        # In full implementation, would use proper graph attention

        return {
            'jobs': job_emb,
            'operations': op_emb,
            'machines': machine_emb,
            'workers': worker_emb,
        }

    def forward(
        self,
        job_features: torch.Tensor,
        op_features: torch.Tensor,
        machine_features: torch.Tensor,
        worker_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass returning global state embedding with message passing.
        """
        # 1. Initial encoding
        embeddings = self.encode(job_features, op_features, machine_features, worker_features)
        
        job_emb = embeddings['jobs']
        op_emb = embeddings['operations']
        m_emb = embeddings['machines']
        w_emb = embeddings['workers']

        # 2. Cross-entity message passing via attention
        # Concatenate all nodes into one sequence: (batch, total_nodes, output_dim)
        combined = torch.cat([job_emb, op_emb, m_emb, w_emb], dim=1)
        
        # Apply self-attention (all nodes attend to all other nodes)
        attn_out, _ = self.attention(combined, combined, combined)
        
        # 3. Informative pooling
        # Split back to entity types
        n_jobs = job_emb.size(1)
        n_ops = op_emb.size(1)
        n_machines = m_emb.size(1)
        # n_workers = w_emb.size(1)
        
        job_final = attn_out[:, :n_jobs, :].mean(dim=1)
        op_final = attn_out[:, n_jobs:n_jobs+n_ops, :].mean(dim=1)
        m_final = attn_out[:, n_jobs+n_ops:n_jobs+n_ops+n_machines, :].mean(dim=1)
        w_final = attn_out[:, n_jobs+n_ops+n_machines:, :].mean(dim=1)

        # Concatenate entity-level summaries
        all_emb = torch.cat([job_final, op_final, m_final, w_final], dim=-1)

        return self.fusion(all_emb)


class JobAgentNetwork(nn.Module):
    """
    Policy network for Job Agent using Pointer Network logic.
    Supports variable job counts via attention.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
    ):
        if not TORCH_AVAILABLE:
            return

        super().__init__()

        self.query_proj = nn.Linear(embed_dim, hidden_dim)
        self.key_proj = nn.Linear(embed_dim, hidden_dim)
        
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        global_state: torch.Tensor,
        job_embeddings: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using Pointer attention.
        
        Args:
            global_state: (batch, embed_dim) summary of entire shop floor
            job_embeddings: (batch, n_jobs, embed_dim) individual job nodes
            action_mask: (batch, n_jobs)
        """
        # 1. Compute attention scores (Pointer weights)
        query = self.query_proj(global_state).unsqueeze(1) # (B, 1, H)
        keys = self.key_proj(job_embeddings) # (B, N, H)
        
        # Dot product attention
        logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) # (B, N)

        # 2. Apply action mask
        if action_mask is not None:
            logits = logits + (1 - action_mask) * (-1e9)

        # 3. Value estimation
        value = self.critic(global_state)

        return logits, value

    def get_action(
        self,
        global_state: torch.Tensor,
        job_embeddings: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, value = self.forward(global_state, job_embeddings, action_mask)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
            log_prob = torch.zeros_like(action, dtype=torch.float32)
        else:
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob


class MachineAgentNetwork(nn.Module):
    """
    Policy network for Machine Agent

    Selects machine mode and validates machine assignment.

    Args:
        state_dim: Dimension of state features
        n_machines: Number of machines
        n_modes: Maximum modes per machine
        hidden_dim: Hidden layer dimension
    """

    def __init__(
        self,
        state_dim: int = 64,
        n_machines: int = 10,
        n_modes: int = 4,
        hidden_dim: int = 128,
    ):
        if not TORCH_AVAILABLE:
            return

        super().__init__()

        self.n_machines = n_machines
        self.n_modes = n_modes

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Machine selection
        self.machine_selector = nn.Linear(hidden_dim, n_machines)

        # Mode selection
        self.mode_selector = nn.Linear(hidden_dim, n_modes)

        # Critic
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        machine_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Returns:
            machine_logits, mode_logits, value
        """
        hidden = self.encoder(state)

        machine_logits = self.machine_selector(hidden)
        mode_logits = self.mode_selector(hidden)
        value = self.critic(hidden)

        if machine_mask is not None:
            machine_logits = machine_logits + (1 - machine_mask) * (-1e9)

        return machine_logits, mode_logits, value

    def get_action(
        self,
        state: torch.Tensor,
        machine_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions

        Returns:
            machine_action, mode_action, log_prob
        """
        machine_logits, mode_logits, value = self.forward(state, machine_mask)

        if deterministic:
            machine_action = torch.argmax(machine_logits, dim=-1)
            mode_action = torch.argmax(mode_logits, dim=-1)
            log_prob = torch.zeros_like(machine_action, dtype=torch.float32)
        else:
            machine_probs = F.softmax(machine_logits, dim=-1)
            mode_probs = F.softmax(mode_logits, dim=-1)

            machine_dist = torch.distributions.Categorical(machine_probs)
            mode_dist = torch.distributions.Categorical(mode_probs)

            machine_action = machine_dist.sample()
            mode_action = mode_dist.sample()

            log_prob = machine_dist.log_prob(machine_action) + mode_dist.log_prob(mode_action)

        return machine_action, mode_action, log_prob


class WorkerAgentNetwork(nn.Module):
    """
    Policy network for Worker Agent

    Selects worker for operation assignment.

    Args:
        state_dim: Dimension of state features
        n_workers: Number of workers
        hidden_dim: Hidden layer dimension
    """

    def __init__(
        self,
        state_dim: int = 64,
        n_workers: int = 10,
        hidden_dim: int = 128,
    ):
        if not TORCH_AVAILABLE:
            return

        super().__init__()

        self.n_workers = n_workers

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.worker_selector = nn.Linear(hidden_dim, n_workers)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        worker_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        hidden = self.encoder(state)

        worker_logits = self.worker_selector(hidden)
        value = self.critic(hidden)

        if worker_mask is not None:
            worker_logits = worker_logits + (1 - worker_mask) * (-1e9)

        return worker_logits, value

    def get_action(
        self,
        state: torch.Tensor,
        worker_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action"""
        worker_logits, value = self.forward(state, worker_mask)

        if deterministic:
            worker_action = torch.argmax(worker_logits, dim=-1)
            log_prob = torch.zeros_like(worker_action, dtype=torch.float32)
        else:
            probs = F.softmax(worker_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            worker_action = dist.sample()
            log_prob = dist.log_prob(worker_action)

        return worker_action, log_prob


class MultiAgentPPO:
    """
    Multi-Agent PPO Trainer for SFJSSP with Graph Encoding.
    """

    def __init__(
        self,
        job_feature_dim: int = 6,
        op_feature_dim: int = 6,
        machine_feature_dim: int = 4,
        worker_feature_dim: int = 4,
        embed_dim: int = 64,
        n_machines: int = 10,
        n_workers: int = 10,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        device: str = 'cpu',
    ):
        if not TORCH_AVAILABLE:
            self.available = False
            return

        self.available = True
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.device = torch.device(device)

        # 1. Central Graph Encoder
        self.encoder = SFJSSPGraphEncoder(
            job_feature_dim=job_feature_dim,
            op_feature_dim=op_feature_dim,
            machine_feature_dim=machine_feature_dim,
            worker_feature_dim=worker_feature_dim,
            output_dim=embed_dim
        ).to(self.device)

        # 2. Agent networks
        self.job_agent = JobAgentNetwork(embed_dim=embed_dim).to(self.device)
        
        self.machine_agent = MachineAgentNetwork(
            state_dim=embed_dim,
            n_machines=n_machines,
        ).to(self.device)
        
        self.worker_agent = WorkerAgentNetwork(
            state_dim=embed_dim,
            n_workers=n_workers,
        ).to(self.device)

        # Optimizers
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.fusion.parameters()}, # Only tune fusion for now
            {'params': self.job_agent.parameters()},
            {'params': self.machine_agent.parameters()},
            {'params': self.worker_agent.parameters()},
        ], lr=lr)

        self.buffer = []

    def select_actions(
        self,
        obs: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Select actions using graph-encoded state"""
        # Convert to tensors
        j_feat = torch.FloatTensor(obs['job_nodes']).unsqueeze(0).to(self.device)
        o_feat = torch.FloatTensor(obs['op_nodes']).unsqueeze(0).to(self.device)
        m_feat = torch.FloatTensor(obs['machine_nodes']).unsqueeze(0).to(self.device)
        w_feat = torch.FloatTensor(obs['worker_nodes']).unsqueeze(0).to(self.device)
        
        mask = obs.get('action_mask')
        # Job mask: valid if any machine/worker/mode is valid for any op of this job
        job_mask = torch.FloatTensor(mask.sum(axis=(1, 2, 3, 4)) > 0).unsqueeze(0).to(self.device)

        # 1. Encode
        node_embs = self.encoder.encode(j_feat, o_feat, m_feat, w_feat)
        global_state = self.encoder.forward(j_feat, o_feat, m_feat, w_feat)

        # 2. Job Selection (Pointer)
        job_action, job_log_prob = self.job_agent.get_action(
            global_state, node_embs['jobs'], job_mask, deterministic
        )

        # 3. Resource Selection (Standard MLP for now, conditioned on global state)
        # Note: In a 10/10 model, machine/worker would also be Pointer-based
        machine_action, mode_action, mac_log_prob = self.machine_agent.get_action(
            global_state, None, deterministic
        )

        worker_action, wrk_log_prob = self.worker_agent.get_action(
            global_state, None, deterministic
        )

        return {
            'job_action': job_action,
            'machine_action': machine_action,
            'mode_action': mode_action,
            'worker_action': worker_action,
            'job_log_prob': job_log_prob,
            'machine_log_prob': mac_log_prob,
            'worker_log_prob': wrk_log_prob,
            'states': {
                'j': j_feat, 'o': o_feat, 'm': m_feat, 'w': w_feat
            }
        }

    def store_transition(
        self,
        states: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
        next_states: Dict[str, torch.Tensor],
        dones: torch.Tensor,
    ):
        """Store transition in replay buffer"""
        self.buffer.append({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
        })

    def update(self, n_epochs: int = 10, batch_size: int = 64):
        """
        Update agents using PPO with formal graph encoding.
        """
        if len(self.buffer) < batch_size:
            return

        # 1. Convert buffer to tensors
        j_feats = torch.cat([t['states']['j'] for t in self.buffer]).to(self.device)
        o_feats = torch.cat([t['states']['o'] for t in self.buffer]).to(self.device)
        m_feats = torch.cat([t['states']['m'] for t in self.buffer]).to(self.device)
        w_feats = torch.cat([t['states']['w'] for t in self.buffer]).to(self.device)
        
        rewards = torch.cat([t['rewards'] for t in self.buffer]).to(self.device)
        dones = torch.cat([t['dones'] for t in self.buffer]).to(self.device)
        
        job_actions = torch.stack([t['actions']['job_action'] for t in self.buffer]).to(self.device).squeeze()
        mac_actions = torch.stack([t['actions']['machine_action'] for t in self.buffer]).to(self.device).squeeze()
        wrk_actions = torch.stack([t['actions']['worker_action'] for t in self.buffer]).to(self.device).squeeze()
        
        old_job_log_probs = torch.stack([t['actions']['job_log_prob'] for t in self.buffer]).to(self.device).detach().squeeze()
        old_mac_log_probs = torch.stack([t['actions']['machine_log_prob'] for t in self.buffer]).to(self.device).detach().squeeze()
        old_wrk_log_probs = torch.stack([t['actions']['worker_log_prob'] for t in self.buffer]).to(self.device).detach().squeeze()

        # 2. Compute returns and advantages
        with torch.no_grad():
            node_embs = self.encoder.encode(j_feats, o_feats, m_feats, w_feats)
            global_state = self.encoder.forward(j_feats, o_feats, m_feats, w_feats)
            
            _, v_job = self.job_agent(global_state, node_embs['jobs'])
            _, _, v_mac = self.machine_agent(global_state)
            _, v_wrk = self.worker_agent(global_state)
            
            returns = []
            discounted_reward = 0
            for r, d in zip(reversed(rewards.tolist()), reversed(dones.tolist())):
                if d: discounted_reward = 0
                discounted_reward = r + (self.gamma * discounted_reward)
                returns.insert(0, discounted_reward)
            
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            job_adv = returns - v_job.squeeze()
            mac_adv = returns - v_mac.squeeze()
            wrk_adv = returns - v_wrk.squeeze()

        # 3. PPO Update Epochs
        for _ in range(n_epochs):
            # Encode for each epoch (shared encoder)
            node_embs = self.encoder.encode(j_feats, o_feats, m_feats, w_feats)
            global_state = self.encoder.forward(j_feats, o_feats, m_feats, w_feats)

            # Job Agent Update
            logits_j, val_j = self.job_agent(global_state, node_embs['jobs'])
            dist_j = torch.distributions.Categorical(F.softmax(logits_j, dim=-1))
            new_log_probs_j = dist_j.log_prob(job_actions)
            ratio_j = torch.exp(new_log_probs_j - old_job_log_probs)
            surr1_j = ratio_j * job_adv
            surr2_j = torch.clamp(ratio_j, 1-self.clip_epsilon, 1+self.clip_epsilon) * job_adv
            j_loss = -torch.min(surr1_j, surr2_j).mean() + 0.5 * F.mse_loss(val_j.squeeze(), returns)
            
            # Machine Agent Update
            logits_m, _, val_m = self.machine_agent(global_state)
            dist_m = torch.distributions.Categorical(F.softmax(logits_m, dim=-1))
            new_log_probs_m = dist_m.log_prob(mac_actions)
            ratio_m = torch.exp(new_log_probs_m - old_mac_log_probs)
            surr1_m = ratio_m * mac_adv
            surr2_m = torch.clamp(ratio_m, 1-self.clip_epsilon, 1+self.clip_epsilon) * mac_adv
            m_loss = -torch.min(surr1_m, surr2_m).mean() + 0.5 * F.mse_loss(val_m.squeeze(), returns)
            
            # Worker Agent Update
            logits_w, val_w = self.worker_agent(global_state)
            dist_w = torch.distributions.Categorical(F.softmax(logits_w, dim=-1))
            new_log_probs_w = dist_w.log_prob(wrk_actions)
            ratio_w = torch.exp(new_log_probs_w - old_wrk_log_probs)
            surr1_w = ratio_w * wrk_adv
            surr2_w = torch.clamp(ratio_w, 1-self.clip_epsilon, 1+self.clip_epsilon) * wrk_adv
            w_loss = -torch.min(surr1_w, surr2_w).mean() + 0.5 * F.mse_loss(val_w.squeeze(), returns)
            
            total_loss = j_loss + m_loss + w_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        self.buffer = []

    def save(self, path: str):
        """Save model checkpoints"""
        if not TORCH_AVAILABLE:
            return
        torch.save({
            'job_agent': self.job_agent.state_dict(),
            'machine_agent': self.machine_agent.state_dict(),
            'worker_agent': self.worker_agent.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model checkpoints"""
        if not TORCH_AVAILABLE:
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.job_agent.load_state_dict(checkpoint['job_agent'])
        self.machine_agent.load_state_dict(checkpoint['machine_agent'])
        self.worker_agent.load_state_dict(checkpoint['worker_agent'])


# Fallback implementations when torch is not available
class TorchStub:
    """Stub class when PyTorch is not available"""

    def __init__(self):
        self.available = False

    def select_actions(self, *args, **kwargs):
        raise ImportError("PyTorch is required for DRL agents")

    def update(self, *args, **kwargs):
        raise ImportError("PyTorch is required for DRL agents")
