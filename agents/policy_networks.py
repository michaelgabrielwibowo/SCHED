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


BaseTorchModule = nn.Module if TORCH_AVAILABLE else object


if TORCH_AVAILABLE:
    def _safe_categorical_from_logits(
        logits: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Categorical:
        """Build a numerically stable categorical distribution from masked logits."""
        logits = torch.nan_to_num(logits, nan=-1e9, neginf=-1e9, posinf=1e9)

        if action_mask is not None:
            mask = (action_mask > 0).to(logits.dtype)
            logits = logits + (1 - mask) * (-1e9)
        else:
            mask = torch.ones_like(logits)

        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0) * mask

        totals = probs.sum(dim=-1, keepdim=True)
        fallback = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        probs = torch.where(totals > 0, probs / totals.clamp_min(1e-12), fallback)

        return torch.distributions.Categorical(probs)


class SFJSSPGraphEncoder(BaseTorchModule):
    """
    Encoder for SFJSSP state representation using connectivity-masked attention.

    Encodes entities (Jobs, Operations, Machines, Workers) using global self-attention
    that is strictly masked by the problem's physical and logical connectivity 
    (provided via the adjacency matrix).

    Evidence: Entity-level representation confirmed from DRL scheduling literature

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
        super().__init__()
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
        adjacency: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with connectivity-aware attention.
        
        Args:
            job_features: (batch, n_jobs, job_feature_dim)
            op_features: (batch, n_ops, op_feature_dim)
            machine_features: (batch, n_machines, machine_feature_dim)
            worker_features: (batch, n_workers, worker_feature_dim)
            adjacency: (batch, total_nodes, total_nodes) - Mask for connectivity
            padding_mask: (batch, total_nodes) - 1 for real, 0 for padding
        """
        # 1. Initial encoding
        embeddings = self.encode(job_features, op_features, machine_features, worker_features)
        
        job_emb = embeddings['jobs']
        op_emb = embeddings['operations']
        m_emb = embeddings['machines']
        w_emb = embeddings['workers']

        # 2. Connectivity-aware message passing
        # Concatenate all nodes into one sequence: (batch, total_nodes, output_dim)
        combined = torch.cat([job_emb, op_emb, m_emb, w_emb], dim=1)
        
        # Prepare padding mask for MultiheadAttention (True means MASKED)
        key_padding_mask = None
        if padding_mask is not None:
            key_padding_mask = (padding_mask == 0) # (batch, total_nodes)
            
        # Prepare attention mask from adjacency (0 means MASKED, 1 means ALLOWED)
        attn_mask = None
        if adjacency is not None:
            eye = torch.eye(
                adjacency.size(-1),
                dtype=adjacency.dtype,
                device=adjacency.device,
            ).unsqueeze(0)
            adjacency = torch.maximum(adjacency, eye)

            # MultiheadAttention attn_mask: (L, S) or (N*num_heads, L, S)
            # Use a boolean mask so its dtype matches key_padding_mask.
            base_mask = adjacency == 0

            # Repeat for each attention head: (B, L, S) -> (B*H, L, S)
            num_heads = self.attention.num_heads
            attn_mask = base_mask.repeat_interleave(num_heads, dim=0)
        
        # Apply connectivity-masked self-attention
        attn_out, _ = self.attention(
            combined, combined, combined, 
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )
        
        # 3. Informative pooling (ignoring padded nodes)
        n_jobs = job_emb.size(1)
        n_ops = op_emb.size(1)
        n_machines = m_emb.size(1)
        
        # Pool real nodes only (using padding_mask)
        def masked_mean(tensor, mask):
            # tensor: (B, N, D), mask: (B, N)
            mask = mask.unsqueeze(-1) # (B, N, 1)
            return (tensor * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

        job_final = masked_mean(attn_out[:, :n_jobs, :], padding_mask[:, :n_jobs])
        op_final = masked_mean(attn_out[:, n_jobs:n_jobs+n_ops, :], padding_mask[:, n_jobs:n_jobs+n_ops])
        m_final = masked_mean(attn_out[:, n_jobs+n_ops:n_jobs+n_ops+n_machines, :], padding_mask[:, n_jobs+n_ops:n_jobs+n_ops+n_machines])
        w_final = masked_mean(attn_out[:, n_jobs+n_ops+n_machines:, :], padding_mask[:, n_jobs+n_ops+n_machines:])

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
            dist = _safe_categorical_from_logits(logits, action_mask)
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
            machine_dist = _safe_categorical_from_logits(machine_logits, machine_mask)
            mode_dist = _safe_categorical_from_logits(mode_logits)

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
            dist = _safe_categorical_from_logits(worker_logits, worker_mask)
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
            {'params': self.encoder.parameters()},
            {'params': self.job_agent.parameters()},
            {'params': self.machine_agent.parameters()},
            {'params': self.worker_agent.parameters()},
        ], lr=lr)

        self.buffer = []

    def select_actions(
        self,
        obs: Dict[str, np.ndarray],
        env: Any,  # Need env reference for on-demand resource mask
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Select actions using graph-encoded state and hierarchical masking"""
        # 1. Convert features to tensors
        j_feat = torch.FloatTensor(obs['job_nodes']).unsqueeze(0).to(self.device)
        o_feat = torch.FloatTensor(obs['op_nodes']).unsqueeze(0).to(self.device)
        m_feat = torch.FloatTensor(obs['machine_nodes']).unsqueeze(0).to(self.device)
        w_feat = torch.FloatTensor(obs['worker_nodes']).unsqueeze(0).to(self.device)
        
        job_mask = obs.get('job_mask')
        if job_mask is None:
            job_mask = np.ones(j_feat.shape[1], dtype=np.float32)
        
        # 2. Global encoding
        adj_t = torch.FloatTensor(obs['adjacency']).unsqueeze(0).to(self.device)
        pad_t = torch.FloatTensor(obs['padding_mask']).unsqueeze(0).to(self.device)
        
        node_embs = self.encoder.encode(j_feat, o_feat, m_feat, w_feat)
        global_state = self.encoder.forward(j_feat, o_feat, m_feat, w_feat, adjacency=adj_t, padding_mask=pad_t)

        # 3. Job Selection (Pointer)
        job_mask_t = torch.FloatTensor(job_mask).unsqueeze(0).to(self.device)
        
        job_action, job_log_prob = self.job_agent.get_action(
            global_state, node_embs['jobs'], job_mask_t, deterministic
        )
        valid_jobs = np.flatnonzero(job_mask > 0.0)
        if valid_jobs.size == 0:
            raise RuntimeError("No valid jobs available for PPO action selection.")

        job_idx = int(job_action.item())
        if (
            job_idx >= env.instance.n_jobs
            or job_mask[job_idx] <= 0.0
            or env.instance.get_job(job_idx) is None
        ):
            job_idx = int(valid_jobs[0])
            job_action = torch.tensor([job_idx], device=self.device)

        # 4. Hierarchical Resource Selection
        # Get resource mask for selected job on-demand from environment
        res_mask = env.compute_resource_mask(job_idx) # (n_mac, n_wrk, n_mod)
        
        # Machine mask for selected job
        mac_m = (res_mask.sum(axis=(1, 2)) > 0).astype(np.float32)
        mac_mask_t = torch.FloatTensor(mac_m).unsqueeze(0).to(self.device)
        
        machine_action, mode_action, mac_log_prob = self.machine_agent.get_action(
            global_state, mac_mask_t, deterministic
        )

        # Worker mask for selected job/machine
        valid_machines = np.flatnonzero(mac_m > 0.0)
        if valid_machines.size == 0:
            raise RuntimeError(f"No valid machines for job {job_idx}.")

        mac_idx = int(machine_action.item())
        if mac_idx >= len(mac_m) or mac_m[mac_idx] <= 0.0:
            mac_idx = int(valid_machines[0])
            machine_action = torch.tensor([mac_idx], device=self.device)

        wrk_m = (res_mask[mac_idx].sum(axis=1) > 0).astype(np.float32)
        wrk_mask_t = torch.FloatTensor(wrk_m).unsqueeze(0).to(self.device)

        worker_action, wrk_log_prob = self.worker_agent.get_action(
            global_state, wrk_mask_t, deterministic
        )

        valid_workers = np.flatnonzero(wrk_m > 0.0)
        if valid_workers.size == 0:
            raise RuntimeError(f"No valid workers for job {job_idx} on machine {mac_idx}.")

        wrk_idx = int(worker_action.item())
        if wrk_idx >= len(wrk_m) or wrk_m[wrk_idx] <= 0.0:
            wrk_idx = int(valid_workers[0])
            worker_action = torch.tensor([wrk_idx], device=self.device)

        mode_m = (res_mask[mac_idx, wrk_idx] > 0).astype(np.float32)
        mode_mask_t = torch.FloatTensor(mode_m).unsqueeze(0).to(self.device)
        valid_modes = np.flatnonzero(mode_m > 0.0)
        if valid_modes.size == 0:
            raise RuntimeError(
                f"No valid modes for job {job_idx} on machine {mac_idx} with worker {wrk_idx}."
            )

        mode_idx = int(mode_action.item())
        if mode_idx >= len(mode_m) or mode_m[mode_idx] <= 0.0:
            mode_idx = int(valid_modes[0])
            mode_action = torch.tensor([mode_idx], device=self.device)

        # Determine current op_idx for this job
        job = env.instance.get_job(job_idx)
        op_idx = 0
        for i, op in enumerate(job.operations):
            if not op.is_scheduled:
                op_idx = i
                break

        return {
            'job_action': job_action,
            'op_action': torch.tensor([op_idx], device=self.device),
            'machine_action': machine_action,
            'mode_action': mode_action,
            'worker_action': worker_action,
            'job_log_prob': job_log_prob,
            'machine_log_prob': mac_log_prob,
            'worker_log_prob': wrk_log_prob,
            'states': {
                'j': j_feat, 'o': o_feat, 'm': m_feat, 'w': w_feat,
                'adj': adj_t,
                'pad': pad_t,
                'job_mask': job_mask_t.detach().clone(),
                'machine_mask': mac_mask_t.detach().clone(),
                'worker_mask': wrk_mask_t.detach().clone(),
                'mode_mask': mode_mask_t.detach().clone(),
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

        batch = self._prepare_buffer_tensors()

        # 2. Compute returns and advantages
        with torch.no_grad():
            node_embs, global_state = self._encode_batch(batch)
            
            _, v_job = self.job_agent(global_state, node_embs['jobs'], batch['job_masks'])
            _, _, v_mac = self.machine_agent(global_state, batch['machine_masks'])
            _, v_wrk = self.worker_agent(global_state, batch['worker_masks'])
            
            returns = []
            discounted_reward = 0
            for r, d in zip(reversed(batch['rewards'].tolist()), reversed(batch['dones'].tolist())):
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
            node_embs, global_state = self._encode_batch(batch)
            current = self._compute_current_policy_outputs(batch, node_embs, global_state)

            # Job Agent Update
            ratio_j = torch.exp(current['job_log_probs'] - batch['old_job_log_probs'])
            surr1_j = ratio_j * job_adv
            surr2_j = torch.clamp(ratio_j, 1-self.clip_epsilon, 1+self.clip_epsilon) * job_adv
            j_loss = -torch.min(surr1_j, surr2_j).mean() + 0.5 * F.mse_loss(current['job_values'].squeeze(), returns)
            
            # Machine Agent Update
            ratio_m = torch.exp(current['machine_log_probs'] - batch['old_mac_log_probs'])
            surr1_m = ratio_m * mac_adv
            surr2_m = torch.clamp(ratio_m, 1-self.clip_epsilon, 1+self.clip_epsilon) * mac_adv
            m_loss = -torch.min(surr1_m, surr2_m).mean() + 0.5 * F.mse_loss(current['machine_values'].squeeze(), returns)
            
            # Worker Agent Update
            ratio_w = torch.exp(current['worker_log_probs'] - batch['old_wrk_log_probs'])
            surr1_w = ratio_w * wrk_adv
            surr2_w = torch.clamp(ratio_w, 1-self.clip_epsilon, 1+self.clip_epsilon) * wrk_adv
            w_loss = -torch.min(surr1_w, surr2_w).mean() + 0.5 * F.mse_loss(current['worker_values'].squeeze(), returns)
            
            total_loss = j_loss + m_loss + w_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        self.buffer = []

    def _prepare_buffer_tensors(self) -> Dict[str, torch.Tensor]:
        """Stack buffered transitions into a single PPO batch."""
        def _mask_or_ones(key: str, fallback_width_key: str) -> torch.Tensor:
            tensors = []
            for transition in self.buffer:
                states = transition['states']
                mask = states.get(key)
                if mask is None:
                    width = states[fallback_width_key].shape[1]
                    mask = torch.ones((1, width), dtype=torch.float32)
                tensors.append(mask)
            return torch.cat(tensors).to(self.device)

        return {
            'j_feats': torch.cat([t['states']['j'] for t in self.buffer]).to(self.device),
            'o_feats': torch.cat([t['states']['o'] for t in self.buffer]).to(self.device),
            'm_feats': torch.cat([t['states']['m'] for t in self.buffer]).to(self.device),
            'w_feats': torch.cat([t['states']['w'] for t in self.buffer]).to(self.device),
            'adj_feats': torch.cat([t['states']['adj'] for t in self.buffer]).to(self.device),
            'pad_feats': torch.cat([t['states']['pad'] for t in self.buffer]).to(self.device),
            'job_masks': _mask_or_ones('job_mask', 'j'),
            'machine_masks': _mask_or_ones('machine_mask', 'm'),
            'worker_masks': _mask_or_ones('worker_mask', 'w'),
            'mode_masks': _mask_or_ones('mode_mask', 'm'),
            'rewards': torch.cat([t['rewards'] for t in self.buffer]).to(self.device).view(-1),
            'dones': torch.cat([t['dones'] for t in self.buffer]).to(self.device).view(-1),
            'job_actions': torch.stack([t['actions']['job_action'] for t in self.buffer]).to(self.device).view(-1),
            'mac_actions': torch.stack([t['actions']['machine_action'] for t in self.buffer]).to(self.device).view(-1),
            'mode_actions': torch.stack([t['actions']['mode_action'] for t in self.buffer]).to(self.device).view(-1),
            'wrk_actions': torch.stack([t['actions']['worker_action'] for t in self.buffer]).to(self.device).view(-1),
            'old_job_log_probs': torch.stack([t['actions']['job_log_prob'] for t in self.buffer]).to(self.device).detach().view(-1),
            'old_mac_log_probs': torch.stack([t['actions']['machine_log_prob'] for t in self.buffer]).to(self.device).detach().view(-1),
            'old_wrk_log_probs': torch.stack([t['actions']['worker_log_prob'] for t in self.buffer]).to(self.device).detach().view(-1),
        }

    def _encode_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Encode a prepared PPO batch through the shared graph encoder."""
        node_embs = self.encoder.encode(
            batch['j_feats'],
            batch['o_feats'],
            batch['m_feats'],
            batch['w_feats'],
        )
        global_state = self.encoder.forward(
            batch['j_feats'],
            batch['o_feats'],
            batch['m_feats'],
            batch['w_feats'],
            adjacency=batch['adj_feats'],
            padding_mask=batch['pad_feats'],
        )
        return node_embs, global_state

    def _compute_current_policy_outputs(
        self,
        batch: Dict[str, torch.Tensor],
        node_embs: Dict[str, torch.Tensor],
        global_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Rebuild current action log-probs using the same masks used at selection time."""
        logits_j, val_j = self.job_agent(global_state, node_embs['jobs'], batch['job_masks'])
        dist_j = _safe_categorical_from_logits(logits_j, batch['job_masks'])

        logits_m, logits_mode, val_m = self.machine_agent(global_state, batch['machine_masks'])
        dist_m = _safe_categorical_from_logits(logits_m, batch['machine_masks'])

        # Mode selection is still unmasked at action-selection time; keep PPO
        # consistent with that contract until the selection policy itself changes.
        dist_mode = _safe_categorical_from_logits(logits_mode)

        logits_w, val_w = self.worker_agent(global_state, batch['worker_masks'])
        dist_w = _safe_categorical_from_logits(logits_w, batch['worker_masks'])

        return {
            'job_log_probs': dist_j.log_prob(batch['job_actions']),
            'job_values': val_j,
            'machine_log_probs': (
                dist_m.log_prob(batch['mac_actions']) + dist_mode.log_prob(batch['mode_actions'])
            ),
            'machine_values': val_m,
            'worker_log_probs': dist_w.log_prob(batch['wrk_actions']),
            'worker_values': val_w,
        }

    def save(self, path: str):
        """Save model checkpoints"""
        if not TORCH_AVAILABLE:
            return
        torch.save({
            'encoder': self.encoder.state_dict(),
            'job_agent': self.job_agent.state_dict(),
            'machine_agent': self.machine_agent.state_dict(),
            'worker_agent': self.worker_agent.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model checkpoints"""
        if not TORCH_AVAILABLE:
            return
        checkpoint = torch.load(path, map_location=self.device)
        if 'encoder' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder'])
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
