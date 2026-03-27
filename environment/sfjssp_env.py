"""
SFJSSP Environment - OpenAI Gym Interface

Evidence Status:
- Gym interface: Standard RL interface [CONFIRMED]
- State representation: PROPOSED heterogeneous graph
- Action masking: CONFIRMED from DRL scheduling literature
- Reward design: PROPOSED multi-objective combination
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import copy

from sfjssp_model.instance import SFJSSPInstance, InstanceType
from sfjssp_model.schedule import Schedule
from sfjssp_model.job import Job, Operation
from sfjssp_model.machine import Machine, MachineState
from sfjssp_model.worker import Worker, WorkerState


@dataclass
class SFJSSPAction:
    """
    Action tuple for SFJSSP

    Evidence: Joint action space for dual-resource scheduling [PROPOSED]
    """
    job_id: int  # Which job to schedule
    op_id: int   # Which operation of that job
    machine_id: int  # Which machine to use
    worker_id: int   # Which worker to use
    mode_id: int     # Which machine mode to use


@dataclass
class SFJSSPObservation:
    """
    Environment observation (state)

    Evidence: State must capture all relevant information for decision [CONFIRMED]
    """
    # Current time
    current_time: float

    # Job states
    job_features: np.ndarray  # (n_jobs, n_job_features)
    operation_features: np.ndarray  # (n_ops, n_op_features)

    # Machine states
    machine_features: np.ndarray  # (n_machines, n_machine_features)

    # Worker states
    worker_features: np.ndarray  # (n_workers, n_worker_features)

    # Eligibility masks
    job_eligible_machines: np.ndarray  # (n_ops, n_machines)
    job_eligible_workers: np.ndarray  # (n_ops, n_workers)

    # Action mask (for invalid action filtering)
    action_mask: np.ndarray  # Depends on action space structure

    # Dynamic event info
    pending_arrivals: int = 0
    broken_machines: List[int] = field(default_factory=list)
    absent_workers: List[int] = field(default_factory=list)


class SFJSSPEnv(gym.Env):
    """
    OpenAI Gym environment for SFJSSP

    Features:
    - Supports both static and dynamic instances
    - Action masking for constraint handling
    - Multi-objective reward function
    - Heterogeneous graph state representation (optional)

    Evidence:
    - Gym interface for DRL [CONFIRMED in DRL scheduling papers]
    - Action masking for constraints [CONFIRMED]
    - Dynamic event injection [CONFIRMED in dynamic FJSSP]
    """

    metadata = {'render_modes': ['text', 'gantt']}

    def __init__(
        self,
        instance: SFJSSPInstance,
        render_mode: Optional[str] = None,
        use_graph_state: bool = False,
        reward_weights: Optional[Dict[str, float]] = None,
        normalize_obs: bool = True,
    ):
        """
        Initialize SFJSSP environment

        Args:
            instance: SFJSSP problem instance
            render_mode: Rendering mode ('text' or 'gantt')
            use_graph_state: Use graph-based state representation
            reward_weights: Weights for multi-objective reward
            normalize_obs: Normalize observation features
        """
        super().__init__()

        self.instance = instance
        self.render_mode = render_mode or 'text'
        self.use_graph_state = use_graph_state
        self.normalize_obs = normalize_obs

        # Default reward weights (can be customized)
        self.reward_weights = reward_weights or {
            'makespan': -1.0,
            'energy': -0.1,
            'tardiness': -0.5,
            'ergonomic': -0.2,
        }

        # Normalization factors (computed during reset)
        self.norm_factors = {}

        # Environment state
        self.current_time = 0.0
        self.schedule = Schedule(instance_id=instance.instance_id)
        self.completed_jobs: List[int] = []
        self.active_operations: Dict[Tuple[int, int], float] = {}  # (job, op) -> end_time

        # Dynamic event state
        self.job_queue: List[Job] = []
        self.event_rng = np.random.default_rng()

        # Tracking
        self.step_count = 0
        self.max_steps = 10000  # Safety limit

        # Define action and observation spaces
        self._define_spaces()

    def _define_spaces(self):
        """Define action and observation spaces"""
        n_jobs = self.instance.n_jobs
        n_ops = self.instance.n_operations
        n_machines = self.instance.n_machines
        n_workers = self.instance.n_workers

        # Maximum modes per machine
        max_modes = max(
            len(m.modes) if m.modes else 1
            for m in self.instance.machines
        ) if self.instance.machines else 1

        # Action space: Discrete choices for job/op/machine/worker/mode
        # Using Dict space for structured actions
        self.action_space = spaces.Dict({
            'job_idx': spaces.Discrete(n_jobs),
            'op_idx': spaces.Discrete(10),  # Max 10 ops per job
            'machine_idx': spaces.Discrete(n_machines),
            'worker_idx': spaces.Discrete(n_workers),
            'mode_idx': spaces.Discrete(max_modes),
        })

        # Observation space: Feature matrices
        # Job features: arrival_time, due_date, weight, progress, is_active, is_completed
        n_job_features = 6
        # Operation features: processing_time, is_ready, is_scheduled, is_completed, start_time, completion_time
        n_op_features = 6
        # Machine features: is_available, current_load, energy_rate, breakdown_state
        n_machine_features = 4
        # Worker features: is_available, efficiency, fatigue, absence_state
        n_worker_features = 4

        if self.use_graph_state:
            # Graph-based observation (for GNN policies)
            self.observation_space = spaces.Dict({
                'job_nodes': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(n_jobs, n_job_features),
                    dtype=np.float32
                ),
                'op_nodes': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(n_ops, n_op_features),
                    dtype=np.float32
                ),
                'machine_nodes': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(n_machines, n_machine_features),
                    dtype=np.float32
                ),
                'worker_nodes': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(n_workers, n_worker_features),
                    dtype=np.float32
                ),
                'adjacency': spaces.Box(
                    low=0, high=1,
                    shape=(n_jobs + n_ops + n_machines + n_workers,
                           n_jobs + n_ops + n_machines + n_workers),
                    dtype=np.float32
                ),
            })
        else:
            # Flat observation vector
            obs_dim = (
                1 +  # Current time
                n_jobs * n_job_features +
                n_ops * n_op_features +
                n_machines * n_machine_features +
                n_workers * n_worker_features +
                n_ops * n_machines +  # Eligibility masks
                n_ops * n_workers
            )
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )

    def _get_observation(self) -> SFJSSPObservation:
        """
        Get current state observation

        Evidence: State must capture scheduling status for decision making [CONFIRMED]
        """
        n_jobs = self.instance.n_jobs
        n_machines = self.instance.n_machines
        n_workers = self.instance.n_workers

        # Count total operations
        n_ops = sum(len(job.operations) for job in self.instance.jobs)

        # Job features
        job_features = np.zeros((n_jobs, 6), dtype=np.float32)
        for i, job in enumerate(self.instance.jobs):
            job_features[i, 0] = job.arrival_time / max(1.0, self.current_time)
            job_features[i, 1] = (job.due_date or 0) / max(1.0, self.current_time + 100)
            job_features[i, 2] = job.weight
            completed_ops = sum(1 for op in job.operations if op.is_completed)
            job_features[i, 3] = completed_ops / max(1, len(job.operations))
            job_features[i, 4] = 1.0 if not job.is_completed and any(
                not op.is_completed for op in job.operations
            ) else 0.0
            job_features[i, 5] = 1.0 if job.is_completed else 0.0

        # Operation features
        op_features = np.zeros((n_ops, 6), dtype=np.float32)
        op_idx = 0
        for job in self.instance.jobs:
            for i, op in enumerate(job.operations):
                op_features[op_idx, 0] = self._get_processing_time(op) / 100.0
                op_features[op_idx, 1] = 1.0 if self._is_operation_ready(op) else 0.0
                op_features[op_idx, 2] = 1.0 if op.is_scheduled else 0.0
                op_features[op_idx, 3] = 1.0 if op.is_completed else 0.0
                op_features[op_idx, 4] = op.start_time if op.start_time else 0.0
                op_features[op_idx, 5] = op.completion_time if op.completion_time else 0.0
                op_idx += 1

        # Machine features
        machine_features = np.zeros((n_machines, 4), dtype=np.float32)
        for i, machine in enumerate(self.instance.machines):
            machine_features[i, 0] = 1.0 if machine.is_available(self.current_time) else 0.0
            # Current load: number of scheduled ops
            if machine.machine_id in self.schedule.machine_schedules:
                sched = self.schedule.machine_schedules[machine.machine_id]
                machine_features[i, 1] = len(sched.operations)
            machine_features[i, 2] = machine.power_processing / 100.0
            machine_features[i, 3] = 1.0 if machine.is_broken else 0.0

        # Worker features
        worker_features = np.zeros((n_workers, 4), dtype=np.float32)
        for i, worker in enumerate(self.instance.workers):
            worker_features[i, 0] = 1.0 if worker.is_available(self.current_time) else 0.0
            worker_features[i, 1] = worker.get_efficiency()
            worker_features[i, 2] = worker.fatigue_current
            worker_features[i, 3] = 1.0 if worker.is_absent else 0.0

        # Eligibility masks
        eligible_machines = np.zeros((n_ops, n_machines), dtype=np.float32)
        eligible_workers = np.zeros((n_ops, n_workers), dtype=np.float32)

        op_idx = 0
        for job in self.instance.jobs:
            for op in job.operations:
                for m_id in op.eligible_machines:
                    if m_id < n_machines:
                        eligible_machines[op_idx, m_id] = 1.0
                for w_id in op.eligible_workers:
                    if w_id < n_workers:
                        eligible_workers[op_idx, w_id] = 1.0
                op_idx += 1

        # Create action mask (filter invalid actions)
        action_mask = self._compute_action_mask()

        obs = SFJSSPObservation(
            current_time=self.current_time,
            job_features=job_features,
            operation_features=op_features,
            machine_features=machine_features,
            worker_features=worker_features,
            job_eligible_machines=eligible_machines,
            job_eligible_workers=eligible_workers,
            action_mask=action_mask,
            broken_machines=[m.machine_id for m in self.instance.machines if m.is_broken],
            absent_workers=[w.worker_id for w in self.instance.workers if w.is_absent],
        )

        return obs

    def _get_processing_time(self, op: Operation) -> float:
        """Get minimum processing time for an operation"""
        min_time = float('inf')
        for machine_times in op.processing_times.values():
            min_time = min(min_time, min(machine_times.values()))
        return min_time if min_time < float('inf') else 10.0

    def _is_operation_ready(self, op: Operation) -> bool:
        """Check if operation is ready (predecessors completed)"""
        return op.op_id == 0 or self.schedule.is_operation_scheduled(
            op.job_id, op.op_id - 1
        )

    def _compute_action_mask(self) -> np.ndarray:
        """
        Compute action mask for invalid action filtering

        Evidence: Action masking for hard constraints [CONFIRMED DRL literature]
        """
        n_jobs = self.instance.n_jobs
        n_ops = 10  # Max ops per job
        n_machines = self.instance.n_machines
        n_workers = self.instance.n_workers

        max_modes = max(
            len(m.modes) if m.modes else 1
            for m in self.instance.machines
        ) if self.instance.machines else 1

        # Mask shape matches action space
        mask = np.ones(
            (n_jobs, n_ops, n_machines, n_workers, max_modes),
            dtype=np.float32
        )

        # Mask invalid job/op combinations
        op_idx = 0
        for job in self.instance.jobs:
            for i, op in enumerate(job.operations):
                # Mask if already scheduled or completed
                if op.is_scheduled or op.is_completed:
                    mask[job.job_id, i, :, :, :] = 0.0

                # Mask if not ready (predecessors not done)
                if not self._is_operation_ready(op):
                    mask[job.job_id, i, :, :, :] = 0.0

                # Mask ineligible machines
                for m in range(n_machines):
                    if m not in op.eligible_machines:
                        mask[job.job_id, i, m, :, :] = 0.0

                # Mask ineligible workers
                for w in range(n_workers):
                    if w not in op.eligible_workers:
                        mask[job.job_id, i, :, w, :] = 0.0

                op_idx += 1

        # Mask unavailable machines
        for m in self.instance.machines:
            if not m.is_available(self.current_time):
                mask[:, :, m.machine_id, :, :] = 0.0

        # Mask unavailable workers
        for w in self.instance.workers:
            if not w.is_available(self.current_time):
                mask[:, :, :, w.worker_id, :] = 0.0

        return mask

    def _validate_action(self, action: SFJSSPAction) -> bool:
        """Validate action against constraints"""
        job = self.instance.get_job(action.job_id)
        if job is None:
            return False

        if action.op_id >= len(job.operations):
            return False

        op = job.operations[action.op_id]

        # Check eligibility
        if action.machine_id not in op.eligible_machines:
            return False
        if action.worker_id not in op.eligible_workers:
            return False

        # Check readiness
        if not self._is_operation_ready(op):
            return False

        # Check resource availability
        machine = self.instance.get_machine(action.machine_id)
        worker = self.instance.get_worker(action.worker_id)

        if not machine.is_available(self.current_time):
            return False
        if not worker.is_available(self.current_time):
            return False

        return True

    def _execute_action(self, action: SFJSSPAction) -> float:
        """
        Execute action and return reward

        Returns:
            float: Immediate reward
        """
        job = self.instance.get_job(action.job_id)
        op = job.operations[action.op_id]
        machine = self.instance.get_machine(action.machine_id)
        worker = self.instance.get_worker(action.worker_id)

        # Calculate start time (must be after predecessors and resource availability)
        earliest_start = self.current_time

        # Check predecessor completion
        if op.op_id > 0:
            prev_op = self.schedule.get_operation(action.job_id, op.op_id - 1)
            if prev_op:
                earliest_start = max(earliest_start, prev_op.completion_time)

        # Check machine availability
        if machine.machine_id in self.schedule.machine_schedules:
            sched = self.schedule.machine_schedules[machine.machine_id]
            if sched.operations:
                last_op = sched.operations[-1]
                earliest_start = max(earliest_start, last_op.completion_time)

        # Check worker availability
        worker_available_time = worker.available_time
        if worker.worker_id in self.schedule.worker_schedules:
            sched = self.schedule.worker_schedules[worker.worker_id]
            if sched.operations:
                last_op = sched.operations[-1]
                earliest_start = max(earliest_start, last_op.completion_time)
                worker_available_time = last_op.completion_time

        start_time = earliest_start

        # Calculate and apply rest duration BEFORE calculating processing time
        rest_duration = max(0.0, start_time - worker_available_time)
        if rest_duration > 0:
            worker.record_rest(rest_duration)

        # Get processing time AFTER rest recovery so efficiency improves
        processing_time = op.get_processing_time(
            action.machine_id,
            action.mode_id,
            worker.get_efficiency()
        )

        completion_time = start_time + processing_time

        # Add to schedule
        self.schedule.add_operation(
            job_id=action.job_id,
            op_id=action.op_id,
            machine_id=action.machine_id,
            worker_id=action.worker_id,
            mode_id=action.mode_id,
            start_time=start_time,
            completion_time=completion_time,
            processing_time=processing_time
        )

        # Update operation state
        op.start_time = start_time
        op.completion_time = completion_time
        op.assigned_machine = action.machine_id
        op.assigned_worker = action.worker_id
        op.assigned_mode = action.mode_id
        op.is_scheduled = True

        # Update machine state
        machine.available_time = completion_time
        machine.total_processing_time += processing_time

        # Update worker state
        worker.available_time = completion_time
        worker.record_work(processing_time)

        # Add ergonomic risk
        risk_rate = self.instance.get_ergonomic_risk(action.job_id, action.op_id)
        worker.add_ergonomic_risk(risk_rate, processing_time)

        # Check if job is complete
        if all(o.is_completed for o in job.operations):
            job.is_completed = True
            job.completion_time = completion_time
            self.completed_jobs.append(action.job_id)

        # Update current time
        self.current_time = completion_time

        # Calculate reward
        reward = self._calculate_reward()

        return reward

    def _calculate_reward(self) -> float:
        """
        Calculate multi-objective reward

        Evidence: Multi-objective reward from literature [PROPOSED combination]
        """
        reward = 0.0

        # Evaluate partial schedule
        if self.schedule.scheduled_ops:
            objectives = self.schedule.evaluate(self.instance)

            # Weighted combination
            if 'makespan' in self.reward_weights:
                # Negative makespan (minimize)
                reward += self.reward_weights['makespan'] * objectives.get('makespan', 0)

            if 'energy' in self.reward_weights:
                reward += self.reward_weights['energy'] * objectives.get('total_energy', 0)

            if 'tardiness' in self.reward_weights:
                reward += self.reward_weights['tardiness'] * objectives.get('weighted_tardiness', 0)

            if 'ergonomic' in self.reward_weights:
                reward += self.reward_weights['ergonomic'] * objectives.get('max_ergonomic_exposure', 0)

        # Small penalty for each step to encourage efficiency
        reward -= 0.01

        return reward

    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # All jobs completed
        if len(self.completed_jobs) >= self.instance.n_jobs:
            return True

        # Max steps reached
        if self.step_count >= self.max_steps:
            return True

        # No feasible actions remaining
        mask = self._compute_action_mask()
        if np.sum(mask) == 0:
            return True

        return False

    def step(self, action_dict: Dict[str, int]) -> Tuple[SFJSSPObservation, float, bool, bool, Dict]:
        """
        Execute one environment step

        Args:
            action_dict: Dictionary with keys job_idx, op_idx, machine_idx, worker_idx, mode_idx

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1

        # Convert action dict to SFJSSPAction
        action = SFJSSPAction(
            job_id=action_dict['job_idx'],
            op_id=action_dict['op_idx'],
            machine_id=action_dict['machine_idx'],
            worker_id=action_dict['worker_idx'],
            mode_id=action_dict['mode_idx'],
        )

        # Validate and execute action
        if not self._validate_action(action):
            # Invalid action: large penalty, no state change
            reward = -10.0
            terminated = self._check_termination()
        else:
            reward = self._execute_action(action)
            terminated = self._check_termination()

        # Handle dynamic events
        if self.instance.instance_type == InstanceType.DYNAMIC:
            self._handle_dynamic_events()

        # Get new observation
        observation = self._get_observation()

        # Info dictionary
        info = {
            'step': self.step_count,
            'current_time': self.current_time,
            'completed_jobs': len(self.completed_jobs),
            'total_jobs': self.instance.n_jobs,
            'makespan': self.schedule.makespan,
            'is_feasible': self.schedule.is_feasible,
        }

        if terminated:
            # Final evaluation
            objectives = self.schedule.evaluate(self.instance)
            info['objectives'] = objectives
            info['energy_breakdown'] = self.schedule.energy_breakdown
            info['constraint_violations'] = self.schedule.constraint_violations

        return observation, reward, terminated, False, info

    def _handle_dynamic_events(self):
        """Handle dynamic events (job arrivals, breakdowns)"""
        if self.instance.dynamic_params is None:
            return

        # Generate new job arrivals
        new_job = self.instance.generate_dynamic_job(self.current_time, self.event_rng)
        if new_job:
            self.instance.add_job(new_job)
            # Update observation space if needed
            # (In practice, may need to handle variable-sized state)

        # Generate machine breakdowns
        breakdown = self.instance.generate_breakdown_event(self.current_time, self.event_rng)
        if breakdown:
            machine_id, _, repair_duration = breakdown
            machine = self.instance.get_machine(machine_id)
            if machine:
                machine.schedule_breakdown(self.current_time, repair_duration)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[SFJSSPObservation, Dict]:
        """
        Reset environment to initial state

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            observation, info
        """
        super().reset(seed=seed)

        if seed is not None:
            self.event_rng = np.random.default_rng(seed)

        # Reset instance
        self.instance.reset()

        # Reset environment state
        self.current_time = 0.0
        self.schedule = Schedule(instance_id=self.instance.instance_id)
        self.completed_jobs = []
        self.step_count = 0

        # Compute normalization factors
        if self.normalize_obs:
            self._compute_normalization_factors()

        # Get initial observation
        observation = self._get_observation()

        info = {
            'instance_id': self.instance.instance_id,
            'n_jobs': self.instance.n_jobs,
            'n_machines': self.instance.n_machines,
            'n_workers': self.instance.n_workers,
        }

        return observation, info

    def _compute_normalization_factors(self):
        """Compute factors for observation normalization"""
        # Max processing time
        max_pt = 0.0
        for job in self.instance.jobs:
            for op in job.operations:
                min_pt = float('inf')
                for machine_times in op.processing_times.values():
                    min_pt = min(min_pt, min(machine_times.values()))
                max_pt = max(max_pt, min_pt)

        self.norm_factors = {
            'time': max(1.0, max_pt * self.instance.n_operations),
            'processing_time': max(1.0, max_pt),
        }

    def render(self):
        """Render the environment"""
        if self.render_mode == 'text':
            return self._render_text()
        elif self.render_mode == 'gantt':
            return self._render_gantt()
        return None

    def _render_text(self) -> str:
        """Render schedule as text"""
        lines = [
            f"SFJSSP Schedule: {self.instance.instance_id}",
            f"Makespan: {self.schedule.makespan:.2f}",
            f"Feasible: {self.schedule.is_feasible}",
            "",
            "Machine Schedule:",
        ]

        for machine_id, machine_sched in sorted(self.schedule.machine_schedules.items()):
            lines.append(f"  M{machine_id}:")
            for op in machine_sched.operations:
                lines.append(
                    f"    J{op.job_id}.O{op.op_id}: [{op.start_time:.2f} - {op.completion_time:.2f}] "
                    f"(W{op.worker_id})"
                )

        return "\n".join(lines)

    def _render_gantt(self) -> dict:
        """Render schedule as Gantt chart data"""
        return self.schedule.to_gantt_dict()

    def close(self):
        """Clean up environment"""
        pass
