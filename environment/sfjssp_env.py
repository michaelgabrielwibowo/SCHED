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

try:
    from ..sfjssp_model.instance import SFJSSPInstance, InstanceType
    from ..sfjssp_model.schedule import Schedule
    from ..sfjssp_model.job import Job, Operation
    from ..sfjssp_model.machine import Machine, MachineState
    from ..sfjssp_model.worker import Worker, WorkerState
except ImportError:  # pragma: no cover - supports repo-root imports
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
    - Dynamic arrivals/breakdowns: PARTIAL support in this implementation
    """

    metadata = {'render_modes': ['text', 'gantt']}

    # [INDUSTRY 5.0] Resilience constants for dynamic resizing
    MAX_JOBS = 200
    MAX_OPS = 1000
    MAX_MACHINES = 20
    MAX_WORKERS = 20

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

        # [INDUSTRY 5.0] Scalability Check
        if instance.n_jobs > self.MAX_JOBS:
            raise ValueError(f"Instance too large: n_jobs ({instance.n_jobs}) > MAX_JOBS ({self.MAX_JOBS}).")
        n_ops = sum(len(j.operations) for j in instance.jobs)
        if n_ops > self.MAX_OPS:
            raise ValueError(f"Instance too large: n_ops ({n_ops}) > MAX_OPS ({self.MAX_OPS}).")
        if instance.n_machines > self.MAX_MACHINES:
            raise ValueError(f"Instance too large: n_machines ({instance.n_machines}) > MAX_MACHINES ({self.MAX_MACHINES}).")
        if instance.n_workers > self.MAX_WORKERS:
            raise ValueError(f"Instance too large: n_workers ({instance.n_workers}) > MAX_WORKERS ({self.MAX_WORKERS}).")

        # Default reward weights (can be customized)
        self.reward_weights = reward_weights or {
            'makespan': -1.0,
            'energy': -0.1,
            'tardiness': -0.5,
            'ergonomic': -0.2,
        }

        # Normalization factors (computed during reset)
        self.norm_factors = {}

        # Track previous metrics for delta reward
        self.last_metrics: Dict[str, float] = {}

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
        n_jobs_max = self.MAX_JOBS
        n_ops_max = self.MAX_OPS
        n_machines_max = self.MAX_MACHINES
        n_workers_max = self.MAX_WORKERS
        max_modes = 4  # Max modes per machine

        # Multi-discrete action space: [job_idx, machine_idx, worker_idx, mode_idx]
        self.action_space = spaces.Dict({
            'job_idx': spaces.Discrete(n_jobs_max),
            'machine_idx': spaces.Discrete(n_machines_max),
            'worker_idx': spaces.Discrete(n_workers_max),
            'mode_idx': spaces.Discrete(max_modes)
        })

        if not self.use_graph_state:
            # Flat observation space (simplified for basic RL)
            obs_dim = (
                n_jobs_max * 6 +  # Job features
                n_ops_max * 6 +   # Op features
                n_machines_max * 4 + # Machine features
                n_workers_max * 4 +  # Worker features
                1 # Current time
            )
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
        else:
            # Heterogeneous graph observation space
            total_nodes_max = n_jobs_max + n_ops_max + n_machines_max + n_workers_max
            self.observation_space = spaces.Dict({
                'job_nodes': spaces.Box(low=-np.inf, high=np.inf, shape=(n_jobs_max, 6), dtype=np.float32),
                'op_nodes': spaces.Box(low=-np.inf, high=np.inf, shape=(n_ops_max, 6), dtype=np.float32),
                'machine_nodes': spaces.Box(low=-np.inf, high=np.inf, shape=(n_machines_max, 4), dtype=np.float32),
                'worker_nodes': spaces.Box(low=-np.inf, high=np.inf, shape=(n_workers_max, 4), dtype=np.float32),
                'adjacency': spaces.Box(low=0, high=1, shape=(total_nodes_max, total_nodes_max), dtype=np.float32),
                'job_mask': spaces.Box(low=0, high=1, shape=(n_jobs_max,), dtype=np.float32),
                'padding_mask': spaces.Box(low=0, high=1, shape=(total_nodes_max,), dtype=np.float32),
            })

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment"""
        super().reset(seed=seed)
        if seed is not None:
            self.event_rng = np.random.default_rng(seed)

        # Reset model states
        self.instance.reset()
        self.schedule = Schedule(instance_id=self.instance.instance_id)
        self.current_time = 0.0
        self.completed_jobs = []
        self.active_operations = {}
        self.step_count = 0

        # Compute initial normalization factors if needed
        if self.normalize_obs:
            self._compute_normalization_factors()

        # Initialize last metrics
        eval_metrics = self.schedule.evaluate(self.instance)
        self.last_metrics = {
            'makespan': 0.0,
            'energy': 0.0,
            'tardiness': 0.0,
            'ergonomic': 0.0
        }

        return self._get_observation(), {}

    def step(
        self,
        action: Dict[str, int]
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one scheduling step"""
        self.step_count += 1
        
        # 1. Action Validation
        is_valid, reason = self._validate_action(action)
        
        if not is_valid:
            # Handle invalid action (heavy negative reward)
            return self._get_observation(), -10.0, False, False, {'invalid': True, 'reason': reason}

        # 2. Execute Action
        exec_info = self._execute_action(action)
        
        # 3. Process time advancement (if no immediate actions possible)
        self._advance_time()

        # 4. Check completion
        done = len(self.completed_jobs) == self.instance.n_jobs
        truncated = self.step_count >= self.max_steps

        # 5. Compute Reward
        reward = self._compute_reward(exec_info)

        # 6. Get observation and info
        obs = self._get_observation()
        info = {
            'current_time': self.current_time,
            'completed_jobs': len(self.completed_jobs),
            'makespan': self.schedule.makespan,
            'exec_info': exec_info
        }

        return obs, reward, done, truncated, info

    def _validate_action(self, action: Dict[str, int]) -> Tuple[bool, str]:
        """
        Validate the chosen scheduling action.
        """
        job_idx = action['job_idx']
        machine_id = action['machine_idx']
        worker_id = action['worker_idx']
        mode_id = action['mode_idx']

        # Job valid
        if job_idx >= self.instance.n_jobs:
            return False, "Invalid job index"
        
        job = self.instance.get_job(job_idx)
        if job.is_completed:
            return False, "Job already completed"

        # Operation ready
        next_op = next((op for op in job.operations if not op.is_scheduled), None)
        if not next_op:
            return False, "No unscheduled operations for job"
        
        if not self._is_operation_ready(next_op):
            return False, "Operation precedence not satisfied"

        # Machine and Worker eligibility
        if machine_id not in next_op.eligible_machines:
            return False, f"Machine {machine_id} not eligible for operation"
        
        if worker_id not in next_op.eligible_workers:
            return False, f"Worker {worker_id} not eligible for operation"

        # Mode valid
        machine = self.instance.get_machine(machine_id)
        if machine is None:
            return False, "Invalid machine"
        if mode_id not in next_op.processing_times.get(machine_id, {}):
            return False, "Invalid machine mode"

        resource_mask = self.compute_resource_mask(job_idx)
        if (
            machine_id >= resource_mask.shape[0]
            or worker_id >= resource_mask.shape[1]
            or mode_id >= resource_mask.shape[2]
            or resource_mask[machine_id, worker_id, mode_id] <= 0
        ):
            return False, "Resource assignment is not temporally feasible"

        return True, ""

    def _execute_action(self, action: Dict[str, int]) -> Dict[str, Any]:
        """Execute valid action and update environment state"""
        job_idx = action['job_idx']
        machine_id = action['machine_idx']
        worker_id = action['worker_idx']
        mode_id = action['mode_idx']

        job = self.instance.get_job(job_idx)
        op = next(op for op in job.operations if not op.is_scheduled)
        machine = self.instance.get_machine(machine_id)
        worker = self.instance.get_worker(worker_id)

        # 1. Determine actual start time (projection)
        # Precedence constraint
        prev_ready_time = job.arrival_time
        transport_time = 0.0
        if op.op_id > 0:
            prev_op = self.schedule.get_operation(job_idx, op.op_id - 1)
            transport_time = getattr(op, 'transport_time', 0.0)
            prev_model_op = job.operations[op.op_id - 1]
            prev_ready_time = (
                prev_op.completion_time
                + transport_time
                + getattr(prev_model_op, 'waiting_time', 0.0)
            )

        # Resource availability
        # Note: We must also account for machine setup gaps
        earliest_start = max(
            self.current_time,
            prev_ready_time,
            machine.available_time + machine.setup_time,
            worker.available_time,
            worker.mandatory_shift_lockout_until,
        )
        
        # 2. Refine start time via Industry 5.0 engines (Gap & Worker validation)
        risk_rate = self.instance.get_ergonomic_risk(job_idx, op.op_id)
        
        # We find the earliest feasible slot using the Worker's validation engine
        # which accounts for shifts, OCRA, and rest rules.
        temp_start = earliest_start
        found = False
        for _ in range(50):
            # Machine Check
            m_valid, m_next = machine.validate_gap(temp_start, machine.setup_time)
            if not m_valid:
                temp_start = max(temp_start, m_next)
                continue
            
            # Worker Check
            # We estimate duration first
            est_pt = op.get_processing_time(
                machine_id,
                mode_id,
                worker.get_efficiency(),
            )
            
            w_valid, w_next = worker.validate_assignment(temp_start, est_pt, risk_rate)
            if not w_valid:
                temp_start = max(temp_start, w_next)
                continue
            
            found = True
            break
        
        if not found:
            # Fallback to next shift if no slot found in current horizon
            temp_start = self.instance.period_clock.period_start(self.instance.period_clock.get_period(temp_start) + 1)

        start_time = temp_start
        
        # 3. Calculate final processing time
        efficiency = worker.get_efficiency()
        processing_time = op.get_processing_time(machine_id, mode_id, efficiency)
        completion_time = start_time + processing_time

        # 4. Update model state
        # Machine tracking
        idle_gap = start_time - machine.available_time
        if idle_gap > 0:
            actual_setup = min(idle_gap, machine.setup_time)
            machine.total_setup_time += actual_setup
            machine.total_idle_time += (idle_gap - actual_setup)
        
        machine.available_time = completion_time
        machine.total_processing_time += processing_time
        if machine.total_processing_time == processing_time:
            machine.startup_count += 1
        machine.degrade_tool(processing_time, mode_id) # INDUSTRY 5.0 Tool Wear
        if transport_time > 0:
            machine.total_transport_time += transport_time
        
        # Worker tracking
        worker.available_time = completion_time
        worker.record_work(
            processing_time,
            risk_rate,
            start_time,
            operation_type=op.op_id,
        )
        
        # Operation state
        op.is_scheduled = True
        op.start_time = start_time
        op.completion_time = completion_time
        op.assign_period_bounds(self.instance.period_clock)
        
        # Schedule recording
        self.schedule.add_operation(
            job_id=job_idx,
            op_id=op.op_id,
            machine_id=machine_id,
            worker_id=worker_id,
            mode_id=mode_id,
            start_time=start_time,
            completion_time=completion_time,
            processing_time=processing_time,
            setup_time=actual_setup if idle_gap > 0 else 0.0,
            transport_time=transport_time,
        )

        # 5. Check job completion
        if op.op_id == len(job.operations) - 1:
            job.is_completed = True
            job.completion_time = completion_time
            self.completed_jobs.append(job_idx)

        return {
            'job_id': job_idx,
            'op_id': op.op_id,
            'start_time': start_time,
            'completion_time': completion_time,
            'processing_time': processing_time
        }

    def _advance_time(self):
        """Advance environment clock to the next decision point if needed"""
        # Find the next time something finishes
        next_event = float('inf')
        
        # Next job arrival?
        for job in self.instance.jobs:
            if not job.operations:
                continue
            first_op = job.operations[0]
            if not first_op.is_scheduled and job.arrival_time > self.current_time:
                next_event = min(next_event, job.arrival_time)

        # Next operation completion?
        for job in self.instance.jobs:
            for op in job.operations:
                if op.is_scheduled and not op.is_completed:
                    if op.completion_time > self.current_time:
                        next_event = min(next_event, op.completion_time)
        
        # Only advance if no jobs are currently "ready" at current_time
        if not any(self._compute_job_mask()):
            if next_event != float('inf'):
                self.current_time = next_event
            else:
                # No active operations, but jobs unscheduled?
                # This usually implies a gap/wait state. Advance by small increment.
                self.current_time += 10.0

        for machine in self.instance.machines:
            machine.repair(self.current_time)
        for worker in self.instance.workers:
            worker.end_absence(self.current_time)

        self._inject_dynamic_events()

        # Mark finished operations as completed
        for job in self.instance.jobs:
            for op in job.operations:
                if op.is_scheduled and not op.is_completed and op.completion_time <= self.current_time:
                    op.is_completed = True

    def _get_observation(self) -> Any:
        """
        Get current state observation as heterogeneous graph.
        """
        n_jobs = self.instance.n_jobs
        n_machines = self.instance.n_machines
        n_workers = self.instance.n_workers
        n_ops = sum(len(job.operations) for job in self.instance.jobs)

        # 1. Node Features (Zero-padded)
        job_features = np.zeros((self.MAX_JOBS, 6), dtype=np.float32)
        for i, job in enumerate(self.instance.jobs):
            if i >= self.MAX_JOBS: break
            job_features[i, 0] = job.arrival_time / 1000.0
            job_features[i, 1] = (job.due_date or 0) / 1000.0
            job_features[i, 2] = job.weight
            completed_ops = sum(1 for op in job.operations if op.is_completed)
            job_features[i, 3] = completed_ops / max(1, len(job.operations))
            job_features[i, 4] = 1.0 if not job.is_completed else 0.0
            job_features[i, 5] = 1.0 if job.is_completed else 0.0

        op_features = np.zeros((self.MAX_OPS, 6), dtype=np.float32)
        op_idx_count = 0
        for job in self.instance.jobs:
            for i, op in enumerate(job.operations):
                if op_idx_count >= self.MAX_OPS: break
                op_features[op_idx_count, 0] = self._get_processing_time(op) / 100.0
                op_features[op_idx_count, 1] = 1.0 if self._is_operation_ready(op) else 0.0
                op_features[op_idx_count, 2] = 1.0 if op.is_scheduled else 0.0
                op_features[op_idx_count, 3] = 1.0 if op.is_completed else 0.0
                op_features[op_idx_count, 4] = (op.start_time or 0.0) / 1000.0
                op_features[op_idx_count, 5] = (op.completion_time or 0.0) / 1000.0
                op_idx_count += 1

        machine_features = np.zeros((self.MAX_MACHINES, 4), dtype=np.float32)
        for i, machine in enumerate(self.instance.machines):
            if i >= self.MAX_MACHINES: break
            machine_features[i, 0] = 1.0 if machine.is_available(self.current_time) else 0.0
            machine_features[i, 1] = machine.available_time / 1000.0
            machine_features[i, 2] = machine.power_processing / 50.0
            machine_features[i, 3] = 1.0 if machine.is_broken else 0.0

        worker_features = np.zeros((self.MAX_WORKERS, 4), dtype=np.float32)
        for i, worker in enumerate(self.instance.workers):
            if i >= self.MAX_WORKERS: break
            worker_features[i, 0] = 1.0 if worker.is_available(self.current_time) else 0.0
            worker_features[i, 1] = worker.get_efficiency()
            worker_features[i, 2] = worker.fatigue_current
            worker_features[i, 3] = 1.0 if worker.is_absent else 0.0

        # 2. Adjacency Matrix (Zero-padded)
        total_nodes_max = self.MAX_JOBS + self.MAX_OPS + self.MAX_MACHINES + self.MAX_WORKERS
        adj = np.zeros((total_nodes_max, total_nodes_max), dtype=np.float32)
        
        # Offsets
        o_off = self.MAX_JOBS
        m_off = self.MAX_JOBS + self.MAX_OPS
        w_off = self.MAX_JOBS + self.MAX_OPS + self.MAX_MACHINES
        
        # [FIX] Mask logical/physical connectivity strictly
        # This is part of the "Graph Connectivity" fix
        op_global_idx = 0
        for i, job in enumerate(self.instance.jobs):
            if i >= self.MAX_JOBS: break
            for j, op in enumerate(job.operations):
                if op_global_idx >= self.MAX_OPS: break
                
                # Job <-> Operations (Bidirectional)
                adj[i, o_off + op_global_idx] = 1.0
                adj[o_off + op_global_idx, i] = 1.0
                
                # Precedence: Op -> Next Op (Directed logically, but kept symmetric for GNN simplicity)
                if j < len(job.operations) - 1 and op_global_idx + 1 < self.MAX_OPS:
                    adj[o_off + op_global_idx, o_off + op_global_idx + 1] = 1.0
                    adj[o_off + op_global_idx + 1, o_off + op_global_idx] = 1.0
                
                # Eligibility: Op <-> Machine
                for m_id in op.eligible_machines:
                    if m_id < self.MAX_MACHINES:
                        adj[o_off + op_global_idx, m_off + m_id] = 1.0
                        adj[m_off + m_id, o_off + op_global_idx] = 1.0
                    
                # Eligibility: Op <-> Worker
                for w_id in op.eligible_workers:
                    if w_id < self.MAX_WORKERS:
                        adj[o_off + op_global_idx, w_off + w_id] = 1.0
                        adj[w_off + w_id, o_off + op_global_idx] = 1.0
                    
                op_global_idx += 1

        # 3. Padding Mask (1 for real, 0 for padding)
        padding_mask = np.zeros(total_nodes_max, dtype=np.float32)
        padding_mask[:n_jobs] = 1.0
        padding_mask[o_off : o_off + n_ops] = 1.0
        padding_mask[m_off : m_off + n_machines] = 1.0
        padding_mask[w_off : w_off + n_workers] = 1.0

        job_mask = np.zeros(self.MAX_JOBS, dtype=np.float32)
        real_job_mask = self._compute_job_mask()
        job_mask[:len(real_job_mask)] = real_job_mask

        if self.normalize_obs and self.norm_factors:
            job_features = job_features / self.norm_factors.get('job', 1.0)
            op_features = op_features / self.norm_factors.get('op', 1.0)
            machine_features = machine_features / self.norm_factors.get('machine', 1.0)
            worker_features = worker_features / self.norm_factors.get('worker', 1.0)

        graph_obs = {
            'job_nodes': job_features,
            'op_nodes': op_features,
            'machine_nodes': machine_features,
            'worker_nodes': worker_features,
            'adjacency': adj,
            'job_mask': job_mask,
            'padding_mask': padding_mask,
        }

        if self.use_graph_state:
            return graph_obs

        flat_obs = np.concatenate([
            job_features.reshape(-1),
            op_features.reshape(-1),
            machine_features.reshape(-1),
            worker_features.reshape(-1),
            np.array([
                self.current_time / self.norm_factors.get('time', 1.0)
                if self.normalize_obs else self.current_time
            ], dtype=np.float32),
        ]).astype(np.float32)
        return flat_obs


    def _compute_job_mask(self) -> np.ndarray:
        """Compute 1D mask for valid jobs"""
        mask = np.zeros(self.instance.n_jobs, dtype=np.float32)
        for i, job in enumerate(self.instance.jobs):
            if i in self.completed_jobs:
                continue
            next_op = next((op for op in job.operations if not op.is_scheduled), None)
            if next_op and self._is_operation_ready(next_op):
                mask[i] = 1.0
        return mask

    def compute_resource_mask(self, job_idx: int) -> np.ndarray:
        """
        Compute valid machines and workers for the current operation of a job.
        Returns a (n_machines, n_workers, max_modes) mask.
        """
        max_modes = 4
        mask = np.zeros((self.instance.n_machines, self.instance.n_workers, max_modes), dtype=np.float32)
        
        job = self.instance.get_job(job_idx)
        next_op = next((op for op in job.operations if not op.is_scheduled), None)
        if not next_op:
            return mask

        prev_comp = 0.0
        if next_op.op_id > 0:
            prev_op = self.schedule.get_operation(job_idx, next_op.op_id - 1)
            if prev_op:
                prev_model_op = job.operations[next_op.op_id - 1]
                prev_comp = (
                    prev_op.completion_time
                    + prev_op.transport_time
                    + getattr(prev_model_op, 'waiting_time', 0.0)
                )
        
        for m_id in next_op.eligible_machines:
            machine = self.instance.get_machine(m_id)
            if machine is None:
                continue
            for w_id in next_op.eligible_workers:
                worker = self.instance.get_worker(w_id)
                if worker is None:
                    continue

                earliest_start = max(
                    self.current_time,
                    job.arrival_time if next_op.op_id == 0 else prev_comp,
                    machine.available_time + machine.setup_time,
                    worker.available_time,
                    worker.mandatory_shift_lockout_until,
                )

                for mode_id in range(max_modes):
                    if mode_id not in next_op.processing_times.get(m_id, {}):
                        continue
                    est_pt = next_op.get_processing_time(
                        m_id,
                        mode_id,
                        worker.get_efficiency(),
                    )
                    temp_start = earliest_start
                    risk_rate = self.instance.get_ergonomic_risk(job_idx, next_op.op_id)
                    for _ in range(50):
                        m_valid, m_next = machine.validate_gap(temp_start, machine.setup_time)
                        if not m_valid:
                            temp_start = max(temp_start, m_next)
                            continue
                        w_valid, w_next = worker.validate_assignment(temp_start, est_pt, risk_rate)
                        if not w_valid:
                            temp_start = max(temp_start, w_next)
                            continue
                        mask[m_id, w_id, mode_id] = 1.0
                        break
        return mask

    def _is_operation_ready(self, op: Operation) -> bool:
        """Check if precedence constraints are met"""
        if op.op_id == 0:
            job = self.instance.get_job(op.job_id)
            return job is not None and job.arrival_time <= self.current_time
        prev_op = self.instance.get_job(op.job_id).operations[op.op_id - 1]
        if not prev_op.is_completed:
            return False
        required_start = (
            prev_op.completion_time
            + getattr(prev_op, 'transport_time', 0.0)
            + getattr(prev_op, 'waiting_time', 0.0)
        )
        return required_start <= self.current_time

    def _get_processing_time(self, op: Operation) -> float:
        """Estimate processing time (averaged across eligible machines)"""
        times = []
        for m_id in op.eligible_machines:
            if m_id in op.processing_times:
                for pt in op.processing_times[m_id].values():
                    times.append(pt)
        return np.mean(times) if times else 50.0

    def _compute_reward(self, exec_info: Dict[str, Any]) -> float:
        """Compute multi-objective reward"""
        eval_metrics = self.schedule.evaluate(self.instance)
        
        # Delta reward (improvement in objectives)
        reward = 0.0
        
        # 1. Makespan penalty (negative of makespan increment)
        ms_delta = eval_metrics['makespan'] - self.last_metrics['makespan']
        reward += self.reward_weights['makespan'] * ms_delta
        
        # 2. Energy penalty
        energy_delta = eval_metrics['total_energy'] - self.last_metrics['energy']
        reward += self.reward_weights['energy'] * (energy_delta / 1000.0)
        
        # 3. Tardiness penalty
        tardiness_delta = eval_metrics['total_tardiness'] - self.last_metrics['tardiness']
        reward += self.reward_weights['tardiness'] * (tardiness_delta / 100.0)
        
        # 4. Ergonomic penalty
        # Penalize if max exposure increased
        ergo_max = eval_metrics['max_ergonomic_exposure']
        if ergo_max > 2.2: # Industry 5.0 threshold
            reward += self.reward_weights['ergonomic'] * (ergo_max - 2.2)

        # Update last metrics
        self.last_metrics = {
            'makespan': eval_metrics['makespan'],
            'energy': eval_metrics['total_energy'],
            'tardiness': eval_metrics['total_tardiness'],
            'ergonomic': ergo_max
        }

        return float(reward)

    def _compute_normalization_factors(self):
        """Compute factors to normalize observation features to [0, 1] range"""
        max_processing = 1.0
        for job in self.instance.jobs:
            for op in job.operations:
                for mode_times in op.processing_times.values():
                    max_processing = max(max_processing, max(mode_times.values(), default=1.0))

        max_power = max(
            [machine.power_processing for machine in self.instance.machines] or [1.0]
        )
        max_worker_scale = max(
            [
                max(worker.base_efficiency, worker.labor_cost_per_hour / 10.0)
                for worker in self.instance.workers
            ] or [1.0]
        )
        self.norm_factors = {
            'time': max(1.0, self.instance.planning_horizon / 100.0),
            'job': max(1.0, self.instance.planning_horizon / 100.0),
            'op': max(1.0, max_processing / 10.0),
            'machine': max(1.0, max_power / 10.0),
            'worker': max(1.0, max_worker_scale),
        }

    def render(self):
        """Render current environment state"""
        if self.render_mode == 'text':
            print(f"Time: {self.current_time:.2f}, Completed Jobs: {len(self.completed_jobs)}/{self.instance.n_jobs}")
            print(f"Makespan: {self.schedule.makespan:.2f}, Total Energy: {self.schedule.energy_breakdown.get('total', 0):.2f}")
        elif self.render_mode == 'gantt':
            return self.schedule.to_gantt_dict()

    def _inject_dynamic_events(self):
        """Inject dynamic arrivals and breakdowns for dynamic instances."""
        if self.instance.instance_type != InstanceType.DYNAMIC or self.instance.dynamic_params is None:
            return

        new_job = self.instance.generate_dynamic_job(self.current_time, self.event_rng)
        if new_job is not None:
            self.instance.add_job(new_job)

        breakdown = self.instance.generate_breakdown_event(self.current_time, self.event_rng)
        if breakdown is not None:
            machine_id, breakdown_time, repair_duration = breakdown
            machine = self.instance.get_machine(machine_id)
            if machine is not None:
                machine.schedule_breakdown(breakdown_time, repair_duration)
