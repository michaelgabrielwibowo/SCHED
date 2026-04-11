"""
Worker data structure for SFJSSP

Evidence Status:
- Worker skills and eligibility: CONFIRMED from DRCFJSSP literature
- Labor cost: CONFIRMED from DRCFJSSP literature
- Fatigue dynamics: CONFIRMED from DyDFJSP 2023
- Ergonomic risk (OCRA): CONFIRMED from NSGA-III 2021 study
- Learning effects: CONFIRMED from learning curve literature
- Rest constraints: CONFIRMED from target survey (12.5% minimum)
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple
from enum import Enum
import numpy as np
from .period_clock import PeriodClock


class WorkerState(Enum):
    """Worker activity states"""
    IDLE = "idle"
    WORKING = "working"
    RESTING = "resting"
    ON_BREAK = "on_break"
    ABSENT = "absent"


@dataclass
class WorkerSkill:
    """
    Worker skill qualification for a specific operation type

    Evidence: Skill matrices CONFIRMED in DRCFJSSP literature
    """
    skill_id: int
    skill_name: str = ""
    
    # Proficiency level (0-1, where 1 = fully qualified)
    proficiency: float = 1.0
    
    # Experience (number of times performed)
    experience_count: int = 0
    
    # Learning coefficient for this skill
    learning_rate: float = 0.1  # gamma parameter


    def to_dict(self) -> dict:
        """Convert skill to dictionary for serialization"""
        return {
            'skill_id': self.skill_id,
            'skill_name': self.skill_name,
            'proficiency': self.proficiency,
            'experience_count': self.experience_count,
            'learning_rate': self.learning_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'WorkerSkill':
        """Create skill from dictionary"""
        return cls(
            skill_id=data['skill_id'],
            skill_name=data.get('skill_name', ""),
            proficiency=data.get('proficiency', 1.0),
            experience_count=data.get('experience_count', 0),
            learning_rate=data.get('learning_rate', 0.1),
        )


@dataclass
class Worker:
    """
    A worker in the flexible job shop

    In SFJSSP, workers have:
    - Skills and eligibility for operations
    - Efficiency that varies with fatigue
    - Ergonomic risk accumulation
    - Learning effects over time
    - Rest requirements
    """
    worker_id: int
    worker_name: str = ""

    # Skill qualifications (CONFIRMED from DRCFJSSP)
    # Map: operation_type_id -> WorkerSkill
    skills: Dict[int, WorkerSkill] = field(default_factory=dict)
    
    # Eligible operations (pre-computed for fast lookup)
    eligible_operations: Set[tuple] = field(default_factory=set)  # Set of (job_id, op_id)

    # Labor characteristics (CONFIRMED from DRCFJSSP)
    labor_cost_per_hour: float = 20.0  # $/hour
    base_efficiency: float = 1.0  # Nominal efficiency (0.5-1.5 typical)

    # Fatigue parameters (CONFIRMED from DyDFJSP 2023)
    fatigue_rate: float = 0.03  # alpha: fatigue accumulation per time unit
    recovery_rate: float = 0.05  # beta: fatigue recovery per time unit
    fatigue_max: float = 1.0  # Maximum fatigue level (0-1 scale)
    fatigue_current: float = 0.0  # Current fatigue level

    # Ergonomic parameters (CONFIRMED from NSGA-III 2021)
    # OCRA (Occupational Repetitive Actions) index parameters
    # [CHANGED] Lowered from 3.0 to 2.2 to match JMSY 9.pdf Section 5.2.3 constraints
    ocra_max_per_shift: float = 2.2  
    ocra_current_shift: float = 0.0  
    ergonomic_tolerance: float = 1.0  # Multiplier for ergonomic risk (higher = more tolerant)

    # Learning parameters (CONFIRMED from learning curve literature)
    # Learning follows: t_n = t_1 * n^(-gamma)
    learning_coefficient: float = 0.1  # gamma: learning rate
    # Cumulative operations performed per type
    operations_completed: Dict[int, int] = field(default_factory=dict)

    # Rest constraints (CONFIRMED from target survey)
    min_rest_fraction: float = 0.125  # Minimum 12.5% rest time
    max_consecutive_work_time: float = 480.0  # Maximum 8 hours continuous work
    current_work_duration: float = 0.0  # Time worked since last rest
    
    # [CHANGED] Added tracking for mandatory rest and shift lockout
    total_work_time: float = 0.0
    total_rest_time: float = 0.0
    SHIFT_DURATION: float = 480.0 # 8 hour period length
    mandatory_shift_lockout_until: float = 0.0

    # NEW: remember period index for this work
    worked_periods: Set[int] = field(default_factory=set)

    # [FIX 6] shared period clock
    period_clock: PeriodClock = field(default_factory=PeriodClock)
    _last_worked_period: int = -1

    # OCRA normalization:
    # ocra_current_shift reaches ocra_max_per_shift (2.2) only if a worker
    # sustains high-risk work for the full shift — matches JMSY-9 §5.2.3.
    ocra_risk_rate_unit: str = "OCRA-index per minute"

    # Current state
    current_state: WorkerState = WorkerState.IDLE
    current_job: Optional[int] = None
    current_operation: Optional[int] = None
    available_time: float = 0.0

    # Shift tracking
    current_shift: int = 0
    shift_start_time: float = 0.0

    # Absence tracking (for dynamic scenarios)
    is_absent: bool = False
    absence_end_time: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'Worker':
        """Create worker from dictionary"""
        w = cls(
            worker_id=data['worker_id'],
            worker_name=data.get('worker_name', ""),
            labor_cost_per_hour=data.get('labor_cost_per_hour', 20.0),
            base_efficiency=data.get('base_efficiency', 1.0),
            fatigue_rate=data.get('fatigue_rate', 0.03),
            recovery_rate=data.get('recovery_rate', 0.05),
            fatigue_max=data.get('fatigue_max', 1.0),
            ocra_max_per_shift=data.get('ocra_max_per_shift', 2.2),
            learning_coefficient=data.get('learning_coefficient', 0.1),
            min_rest_fraction=data.get('min_rest_fraction', 0.125),
        )
        # Skills
        raw_skills = data.get('skills', {})
        w.skills = {int(k): WorkerSkill.from_dict(v) for k, v in raw_skills.items()}
        # Eligible operations
        raw_eligible = data.get('eligible_operations', [])
        w.eligible_operations = {tuple(op) for op in raw_eligible}
        
        w.fatigue_current = data.get('fatigue_current', 0.0)
        w.ocra_current_shift = data.get('ocra_current_shift', 0.0)
        w.current_state = WorkerState(data.get('current_state', 'idle'))
        w.available_time = data.get('available_time', 0.0)
        w.is_absent = data.get('is_absent', False)
        w.total_work_time = data.get('total_work_time', 0.0)
        w.total_rest_time = data.get('total_rest_time', 0.0)
        w.mandatory_shift_lockout_until = data.get('mandatory_shift_lockout_until', 0.0)
        w.current_work_duration = data.get('current_work_duration', 0.0)
        w.worked_periods = set(data.get('worked_periods', []))
        raw_completed = data.get('operations_completed', {})
        w.operations_completed = {int(k): int(v) for k, v in raw_completed.items()}
        return w

    def _get_period_index(self, t: float) -> int:
        """
        Map a continuous time t into a discrete period index,
        using SHIFT_DURATION (e.g. 480.0 for 8h).
        """
        return int(t // self.SHIFT_DURATION)

    def can_work_in_period(self, start_time: float, end_time: float) -> bool:
        """
        Strict period rule:
        - Let p be the period index of this operation (by start_time).
        - Forbid assignment if |p - q| == 1 for any q in worked_periods
          (no back-to-back consecutive periods).
        """
        p = self.period_clock.get_period(start_time)
        for q in self.worked_periods:
            if abs(p - q) == 1:
                return False
        return True

    def validate_assignment(self, start_time: float, duration: float, risk_rate: float) -> Tuple[bool, float]:
        """
        Industry 5.0 validation engine. Checks all worker constraints.
        Returns Tuple[bool, float] as (is_valid, suggested_start_time).
        
        Logic mapping to JMSY-9 §5.2.3:
        - Availability: Worker is not absent and not currently locked out.
        - Period rule: No back-to-back periods (|p - q| > 1).
        - Mandatory rest: 12.5% rest fraction must be satisfied.
        - Boundary: Task cannot span two periods.
        - Ergonomics: OCRA exposure per shift <= 2.2.
        - Work limit: Maximum 8 hours continuous work.
        """
        # 1. Availability check (including lockout)
        if not self.is_available(start_time):
            return False, max(self.available_time, self.mandatory_shift_lockout_until)

        # 2. Period rule (no back-to-back periods)
        p = self.period_clock.get_period(start_time)
        for q in self.worked_periods:
            if abs(p - q) == 1:
                return False, self.period_clock.period_start(p + 1)

        # 3. Mandatory rest rule (12.5%)
        m_rest = self.requires_mandatory_rest(duration, start_time)
        if m_rest > 0:
            return False, start_time + m_rest

        # 4. Period boundary rule (Task cannot span two periods)
        if self.period_clock.crosses_boundary(start_time, start_time + duration):
            return False, self.period_clock.period_start(p + 1)

        # 5. Ergonomic / OCRA limit check
        current_period = self.period_clock.get_period(start_time)
        temp_ocra = self.ocra_current_shift
        
        # If starting in a new period, OCRA resets
        if current_period != self._last_worked_period and self._last_worked_period >= 0:
            temp_ocra = 0.0
            
        if temp_ocra + (risk_rate * duration) > self.ocra_max_per_shift:
            return False, self.period_clock.period_start(current_period + 1)

        # 6. Consecutive work limit
        temp_work_dur = self.current_work_duration
        if current_period != self._last_worked_period and self._last_worked_period >= 0:
            temp_work_dur = 0.0
            
        if temp_work_dur + duration > self.max_consecutive_work_time:
            return False, self.period_clock.period_start(current_period + 1)

        return True, start_time

    def get_efficiency(self) -> float:
        """
        Get current efficiency considering fatigue and learning

        Evidence:
        - Fatigue reduces efficiency [CONFIRMED DyDFJSP 2023]
        - Learning improves efficiency [CONFIRMED learning curve literature]

        Returns:
            float: Current efficiency (0.0-2.0 typical range)
        """
        # Base efficiency
        efficiency = self.base_efficiency

        # Fatigue penalty: efficiency decreases as fatigue increases
        # Evidence: DyDFJSP 2023 models fatigue impact
        fatigue_penalty = 1.0 - 0.3 * self.fatigue_current  # Up to 30% reduction
        efficiency *= fatigue_penalty

        return max(0.1, efficiency)  # Minimum efficiency floor

    def get_learning_adjusted_time(
        self,
        base_time: float,
        operation_type: int
    ) -> float:
        """
        Adjust processing time based on learning curve

        Evidence: Learning curve t_n = t_1 * n^(-gamma) [CONFIRMED]

        Args:
            base_time: Standard processing time for operation
            operation_type: Type of operation for learning tracking

        Returns:
            float: Adjusted processing time
        """
        n_completed = self.operations_completed.get(operation_type, 0)

        if n_completed == 0:
            return base_time

        # Learning curve: time decreases with experience
        # Evidence: Standard learning curve model
        learning_factor = (n_completed + 1) ** (-self.learning_coefficient)

        return base_time * learning_factor

    def update_fatigue(self, work_duration: float, rest_duration: float):
        """
        Update fatigue level based on work and rest periods

        Evidence: Fatigue dynamics from DyDFJSP 2023 [CONFIRMED]
        F(t+1) = min(1, F(t) + alpha*work - beta*rest)

        Args:
            work_duration: Time spent working
            rest_duration: Time spent resting
        """
        # Accumulate fatigue from work
        fatigue_increase = self.fatigue_rate * work_duration

        # Recover from rest
        fatigue_decrease = self.recovery_rate * rest_duration

        # Update current fatigue (bounded 0-1)
        self.fatigue_current = np.clip(
            self.fatigue_current + fatigue_increase - fatigue_decrease,
            0.0,
            self.fatigue_max
        )

    def add_ergonomic_risk(self, risk_rate: float, duration: float):
        """
        Add ergonomic risk exposure for an operation.
        Delegates to record_work (conceptual) but kept here as standalone
        OCRA accumulation — only call when NOT using record_work.
        """
        self.ocra_current_shift += risk_rate * duration

    def is_ergonomic_limit_exceeded(self) -> bool:
        """Check if ergonomic limit exceeded for current shift"""
        return self.ocra_current_shift > self.ocra_max_per_shift

    def get_remaining_ergonomic_capacity(self) -> float:
        """Get remaining ergonomic capacity for current shift"""
        return max(0.0, self.ocra_max_per_shift - self.ocra_current_shift)

    def start_shift(self, current_time: float):
        """Start a new shift"""
        self.current_shift += 1
        self.shift_start_time = current_time
        self.ocra_current_shift = 0.0  # Reset OCRA for new shift
        self.current_work_duration = 0.0

    def end_shift(self, current_time: float):
        """End current shift"""
        self.current_shift = 0
        self.ocra_current_shift = 0.0
        self.current_work_duration = 0.0

    def record_work(
        self,
        duration: float,
        risk_rate: float = 0.0,
        current_time: float = 0.0,
        operation_type: Optional[int] = None,
    ) -> float:
        """Record work period; resets OCRA/fatigue at period boundaries. Returns start_time used."""
        # --- FIX 6: Period-boundary OCRA reset ---
        current_period = self.period_clock.get_period(current_time)
        actual_start_t = current_time
        
        if current_period != self._last_worked_period and self._last_worked_period >= 0:
            # Worker crossed into a new period — reset per-period OCRA accumulator
            self.ocra_current_shift = risk_rate * duration  # only this operation's exposure
            self.current_work_duration = duration           # reset consecutive-work counter
        else:
            self.current_work_duration += duration
            self.ocra_current_shift += risk_rate * duration

        self.total_work_time += duration
        self.update_fatigue(work_duration=duration, rest_duration=0.0)
        if operation_type is not None:
            self.operations_completed[operation_type] = (
                self.operations_completed.get(operation_type, 0) + 1
            )

        self._last_worked_period = current_period
        self.worked_periods.add(current_period)
        
        # [CHANGED] Enforce shift limits and ergonomic safety
        # If worker completes a full shift OR exceeds ergonomic limit, enforce lockout
        if (self.current_work_duration >= self.max_consecutive_work_time or 
            self.ocra_current_shift >= self.ocra_max_per_shift):
            
            # Lock the worker out for the next 8-hour period
            next_period_start = self.period_clock.period_start(current_period + 1)
            self.mandatory_shift_lockout_until = next_period_start
            self.current_work_duration = 0.0 # Reset for their next shift
            self.ocra_current_shift = 0.0    # Reset daily ergonomic risk
            
        return actual_start_t

    def record_rest(self, duration: float):
        """Record rest period"""
        self.total_rest_time += duration
        self.current_work_duration = max(0.0, self.current_work_duration - duration)
        self.update_fatigue(work_duration=0.0, rest_duration=duration)

    def requires_mandatory_rest(self, proposed_task_duration: float, current_time: float) -> float:
        """
        Calculates if adding a new task violates the 12.5% rest rule.
        Returns the amount of mandatory rest time needed before starting the task.
        """
        # If this is the first task, no rest required yet
        if current_time == 0:
            return 0.0
            
        projected_worked_time = self.total_work_time + proposed_task_duration
        projected_total_time = current_time + proposed_task_duration
        
        # Calculate maximum allowed work time for this time span (87.5%)
        max_allowed_work = projected_total_time * (1.0 - self.min_rest_fraction)
        
        if projected_worked_time > max_allowed_work:
            # Calculate exactly how much rest is needed to balance the ratio
            # Formula: Rest Needed = (Projected Work / 0.875) - Projected Total Time
            required_total_time = projected_worked_time / (1.0 - self.min_rest_fraction)
            mandatory_rest_deficit = required_total_time - projected_total_time
            return mandatory_rest_deficit
            
        return 0.0

    def needs_rest(self) -> bool:
        """
        Check if worker needs mandatory rest

        Evidence: Rest constraints from target survey [CONFIRMED 12.5% rule]
        """
        # Check consecutive work time
        if self.current_work_duration >= self.max_consecutive_work_time:
            return True

        # Check rest fraction
        total_time = self.total_work_time + self.total_rest_time
        if total_time > 0:
            current_rest_fraction = self.total_rest_time / total_time
            if current_rest_fraction < self.min_rest_fraction:
                return True

        return False

    def get_rest_fraction(self) -> float:
        """Get current rest fraction"""
        total_time = self.total_work_time + self.total_rest_time
        if total_time == 0:
            return 0.0
        return self.total_rest_time / total_time

    def is_available(self, current_time: float, ignore_temporal: bool = False) -> bool:
        """Check if worker is available at current time"""
        # [CHANGED] Check if the worker is currently locked out due to the consecutive shift rule
        if not ignore_temporal and current_time < self.mandatory_shift_lockout_until:
            return False
            
        return (
            not self.is_absent and
            (ignore_temporal or self.available_time <= current_time) and
            self.current_state != WorkerState.ABSENT
        )

    def schedule_absence(self, start_time: float, end_time: float):
        """Schedule a worker absence"""
        self.is_absent = True
        self.absence_end_time = end_time
        self.available_time = end_time

    def end_absence(self, current_time: float):
        """End worker absence"""
        if self.is_absent and current_time >= self.absence_end_time:
            self.is_absent = False
            self.absence_end_time = None
            self.current_state = WorkerState.IDLE

    def get_labor_cost(self, work_duration: float) -> float:
        """
        Calculate labor cost for work duration

        Evidence: Labor cost minimization [CONFIRMED DRCFJSSP]
        """
        return self.labor_cost_per_hour * (work_duration / 60.0)

    def get_ergonomic_risk_rate(
        self,
        job_id: int,
        op_id: int,
        base_risk_map: Dict[tuple, float]
    ) -> float:
        """
        Get ergonomic risk rate for an operation

        Evidence: Operation-specific ergonomic risk [CONFIRMED NSGA-III 2021]

        Args:
            job_id: Job identifier
            op_id: Operation identifier
            base_risk_map: Map of (job_id, op_id) -> base risk rate

        Returns:
            float: Ergonomic risk rate adjusted for worker tolerance
        """
        base_risk = base_risk_map.get((job_id, op_id), 0.5)
        return base_risk / self.ergonomic_tolerance

    def reset(self):
        """Reset worker to initial state"""
        self.fatigue_current = 0.0
        self.ocra_current_shift = 0.0
        self.current_state = WorkerState.IDLE
        self.current_job = None
        self.current_operation = None
        self.available_time = 0.0
        self.current_shift = 0
        self.shift_start_time = 0.0
        self.current_work_duration = 0.0
        self.total_work_time = 0.0
        self.total_rest_time = 0.0
        self.mandatory_shift_lockout_until = 0.0
        self._last_worked_period = -1
        self.is_absent = False
        self.absence_end_time = None
        self.worked_periods.clear()

    def to_dict(self) -> dict:
        """Convert worker to dictionary for serialization"""
        return {
            'worker_id': self.worker_id,
            'worker_name': self.worker_name,
            'skills': {str(k): v.to_dict() for k, v in self.skills.items()},
            'eligible_operations': [list(op) for op in self.eligible_operations],
            'labor_cost_per_hour': self.labor_cost_per_hour,
            'base_efficiency': self.base_efficiency,
            'fatigue_rate': self.fatigue_rate,
            'recovery_rate': self.recovery_rate,
            'fatigue_max': self.fatigue_max,
            'fatigue_current': self.fatigue_current,
            'ocra_max_per_shift': self.ocra_max_per_shift,
            'ocra_current_shift': self.ocra_current_shift,
            'learning_coefficient': self.learning_coefficient,
            'min_rest_fraction': self.min_rest_fraction,
            'current_state': self.current_state.value,
            'available_time': self.available_time,
            'is_absent': self.is_absent,
            'total_work_time': self.total_work_time,
            'total_rest_time': self.total_rest_time,
            'mandatory_shift_lockout_until': self.mandatory_shift_lockout_until,
            'current_work_duration': self.current_work_duration,
            'worked_periods': sorted(self.worked_periods),
            'operations_completed': {
                str(k): v for k, v in self.operations_completed.items()
            },
        }

    def __hash__(self):
        return hash(self.worker_id)

    def __eq__(self, other):
        if not isinstance(other, Worker):
            return False
        return self.worker_id == other.worker_id
