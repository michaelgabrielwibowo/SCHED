"""
Training Pipeline for SFJSSP DRL

Evidence Status:
- PPO training loop: CONFIRMED from literature
- Application to SFJSSP: PROPOSED
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def parse_observation(obs) -> tuple:
    """
    Parse SFJSSPObservation into state tensors and masks for each agent.
    """
    if not TORCH_AVAILABLE:
        return None, None, None, None, None, None

    # Handle both SFJSSPObservation dataclass and dict formats
    if hasattr(obs, 'job_features'):
        job_features = obs.job_features
        machine_features = obs.machine_features
        worker_features = obs.worker_features
        action_mask = obs.action_mask
    elif isinstance(obs, dict):
        job_features = obs.get('job_features', obs.get('job_nodes', np.zeros((1, 6))))
        machine_features = obs.get('machine_features', obs.get('machine_nodes', np.zeros((1, 4))))
        worker_features = obs.get('worker_features', obs.get('worker_nodes', np.zeros((1, 4))))
        action_mask = obs.get('action_mask', np.ones((1, 1, 1, 1, 1)))
    else:
        return None, None, None, None, None, None

    # Aggregate: mean across entity dimension (Step 6 will fix this in the network)
    job_agg = np.mean(job_features, axis=0) if job_features.shape[0] > 0 else np.zeros(6)
    machine_agg = np.mean(machine_features, axis=0) if machine_features.shape[0] > 0 else np.zeros(4)
    worker_agg = np.mean(worker_features, axis=0) if worker_features.shape[0] > 0 else np.zeros(4)

    # Pad to state_dim=64
    state_dim = 64
    def pad(arr, dim):
        res = np.zeros(dim, dtype=np.float32)
        res[:min(len(arr), dim)] = arr[:min(len(arr), dim)]
        return torch.tensor(res).unsqueeze(0)

    job_state = pad(job_agg, state_dim)
    machine_state = pad(machine_agg, state_dim)
    worker_state = pad(worker_agg, state_dim)

    # Project high-dim mask to per-agent masks
    # action_mask shape: (n_jobs, n_ops, n_machines, n_workers, n_modes)
    
    # 1. Job mask: job is valid if ANY action is possible for it
    job_m = (np.sum(action_mask, axis=(1, 2, 3, 4)) > 0).astype(np.float32)
    job_mask = torch.tensor(job_m).unsqueeze(0)

    # 2. Machine mask: machine is valid if ANY job/op/worker/mode uses it
    mach_m = (np.sum(action_mask, axis=(0, 1, 3, 4)) > 0).astype(np.float32)
    machine_mask = torch.tensor(mach_m).unsqueeze(0)

    # 3. Worker mask: worker is valid if ANY job/op/machine/mode uses it
    work_m = (np.sum(action_mask, axis=(0, 1, 2, 4)) > 0).astype(np.float32)
    worker_mask = torch.tensor(work_m).unsqueeze(0)

    return job_state, machine_state, worker_state, job_mask, machine_mask, worker_mask


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Environment
    instance_size: str = "small"
    n_episodes: int = 1000
    max_steps_per_episode: int = 500

    # PPO hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    # Network architecture
    hidden_dim: int = 128
    state_dim: int = 64

    # Training
    batch_size: int = 64
    n_epochs: int = 10
    update_interval: int = 2048

    # Logging
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50

    # Device
    device: str = "cpu"


class TrainingPipeline:
    """
    Training pipeline for SFJSSP DRL agents

    Coordinates environment, agents, and training loop.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.episode_rewards = []
        self.episode_makespans = []
        self.training_history = []

        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. Training will use random policies.")
            self.agent = None
        else:
            self._initialize_agents()

    def _initialize_agents(self, n_machines: int = 5, n_workers: int = 5):
        """Initialize DRL agents"""
        from agents.policy_networks import MultiAgentPPO

        self.agent = MultiAgentPPO(
            embed_dim=self.config.state_dim,
            n_machines=n_machines,
            n_workers=n_workers,
            lr=self.config.lr,
            gamma=self.config.gamma,
            clip_epsilon=self.config.clip_epsilon,
            device=self.config.device,
        )

    def train(
        self,
        env,
        n_episodes: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Run training loop
        """
        n_episodes = n_episodes or self.config.n_episodes

        if verbose:
            print(f"Starting training for {n_episodes} episodes...")
            print(f"Device: {self.config.device}")
            print(f"PyTorch available: {TORCH_AVAILABLE}")

        # Re-initialize agents with correct environment dimensions
        if TORCH_AVAILABLE:
            self._initialize_agents(
                n_machines=env.instance.n_machines,
                n_workers=env.instance.n_workers
            )

        for episode in range(n_episodes):
            episode_reward = 0.0
            episode_makespan = 0.0

            # Reset environment
            obs, info = env.reset()

            for step in range(self.config.max_steps_per_episode):
                if not TORCH_AVAILABLE:
                    # Random actions (fallback when PyTorch is not available)
                    action = env.action_space.sample()
                    next_obs, reward, terminated, truncated, info = env.step(action)
                else:
                    # [FIX] Direct pass of observation to graph-enabled agent
                    action_dict = self.agent.select_actions(obs, deterministic=False)

                    # [FIX] Use op_action selected by hierarchical mask in agent
                    action = {
                        'job_idx': action_dict['job_action'].item(),
                        'op_idx': action_dict['op_action'].item(),
                        'machine_idx': action_dict['machine_action'].item(),
                        'worker_idx': action_dict['worker_action'].item(),
                        'mode_idx': action_dict['mode_action'].item(),
                    }

                    next_obs, reward, terminated, truncated, info = env.step(action)

                    # Store transition (agent now handles graph state extraction)
                    self.agent.store_transition(
                        states=action_dict['states'],
                        actions=action_dict,
                        rewards=torch.tensor([reward], dtype=torch.float32),
                        next_states=None, # Update logic will re-encode
                        dones=torch.tensor([terminated or truncated], dtype=torch.float32),
                    )

                    # PPO update on interval
                    total_steps = episode * self.config.max_steps_per_episode + step
                    if total_steps % self.config.update_interval == 0 and total_steps > 0:
                        self.agent.update()

                episode_reward += reward

                if 'makespan' in info:
                    episode_makespan = info['makespan']

                # Update observation for next step
                obs = next_obs

                if terminated or truncated:
                    break

            self.episode_rewards.append(episode_reward)
            self.episode_makespans.append(episode_makespan)

            if verbose and episode % self.config.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                avg_makespan = np.mean(self.episode_makespans[-100:]) if len(self.episode_makespans) >= 100 else np.mean(self.episode_makespans)
                print(f"Episode {episode}: reward={avg_reward:.2f}, makespan={avg_makespan:.1f}")

            self.training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'makespan': episode_makespan,
            })

        return {
            'rewards': self.episode_rewards,
            'makespans': self.episode_makespans,
        }

    def save(self, path: str):
        """Save training checkpoint"""
        import json

        # Save config
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'w') as f:
            json.dump({
                'instance_size': self.config.instance_size,
                'lr': self.config.lr,
                'gamma': self.config.gamma,
                'hidden_dim': self.config.hidden_dim,
            }, f, indent=2)

        # Save agent
        if self.agent and TORCH_AVAILABLE:
            self.agent.save(os.path.join(path, "agent.pt"))

        # Save history
        history_path = os.path.join(path, "history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f)

        print(f"Saved checkpoint to {path}")

    def load(self, path: str):
        """Load training checkpoint"""
        import json

        # Load agent
        if self.agent and TORCH_AVAILABLE:
            self.agent.load(os.path.join(path, "agent.pt"))

        # Load history
        history_path = os.path.join(path, "history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)

        print(f"Loaded checkpoint from {path}")


def run_training(
    instance_path: str = None,
    output_dir: str = "training_output",
    n_episodes: int = 100,
):
    """
    Run training on a benchmark instance

    Args:
        instance_path: Path to benchmark JSON file
        output_dir: Output directory for checkpoints
        n_episodes: Number of training episodes
    """
    from sfjssp_model.instance import SFJSSPInstance
    from environment.sfjssp_env import SFJSSPEnv

    # Load or create instance
    if instance_path and os.path.exists(instance_path):
        print(f"Loading instance from {instance_path}")
        with open(instance_path, 'r') as f:
            data = json.load(f)
        from sfjssp_model.instance import SFJSSPInstance
        instance = SFJSSPInstance.from_dict(data)
    else:
        print("Creating new instance...")
        from experiments.generate_benchmarks import BenchmarkGenerator, GeneratorConfig, InstanceSize
        config = GeneratorConfig(size=InstanceSize.SMALL, seed=42)
        generator = BenchmarkGenerator(config)
        instance = generator.generate()

    # Create environment
    env = SFJSSPEnv(instance)

    # Create training config
    config = TrainingConfig(
        n_episodes=n_episodes,
        instance_size="small",
    )

    # Create pipeline
    pipeline = TrainingPipeline(config)

    # Train
    history = pipeline.train(env, verbose=True)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    pipeline.save(output_dir)

    print(f"\nTraining complete!")
    print(f"Final avg reward: {np.mean(history['rewards'][-10:]):.2f}")
    print(f"Final avg makespan: {np.mean(history['makespans'][-10:]):.1f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SFJSSP DRL agents")
    parser.add_argument("--instance", type=str, default=None, help="Path to benchmark instance")
    parser.add_argument("--output", type=str, default="training_output", help="Output directory")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")

    args = parser.parse_args()

    run_training(args.instance, args.output, args.episodes)
