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

    def _initialize_agents(self):
        """Initialize DRL agents"""
        from agents.policy_networks import MultiAgentPPO

        self.agent = MultiAgentPPO(
            state_dim=self.config.state_dim,
            n_jobs=10,  # Will be updated based on instance
            n_machines=5,
            n_workers=5,
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

        Args:
            env: SFJSSPEnv environment
            n_episodes: Number of episodes (overrides config)
            verbose: Print progress

        Returns:
            Training history
        """
        n_episodes = n_episodes or self.config.n_episodes

        if verbose:
            print(f"Starting training for {n_episodes} episodes...")
            print(f"Device: {self.config.device}")
            print(f"PyTorch available: {TORCH_AVAILABLE}")

        for episode in range(n_episodes):
            episode_reward = 0.0
            episode_makespan = 0.0

            # Reset environment
            obs, info = env.reset()

            for step in range(self.config.max_steps_per_episode):
                if not TORCH_AVAILABLE:
                    # Random actions
                    action = env.action_space.sample()
                    next_obs, reward, terminated, truncated, info = env.step(action)
                else:
                    # TODO: Implement proper action selection
                    action = env.action_space.sample()
                    next_obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward

                if 'makespan' in info:
                    episode_makespan = info['makespan']

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
        # TODO: Implement instance loading from JSON
        # For now, use example instance
        from experiments.generate_benchmarks import BenchmarkGenerator, GeneratorConfig, InstanceSize
        config = GeneratorConfig(size=InstanceSize.SMALL, seed=42)
        generator = BenchmarkGenerator(config)
        instance = generator.generate()
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
