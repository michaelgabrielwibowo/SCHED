"""
Machine and Machine Mode data structures for SFJSSP

Evidence Status:
- Basic machine structure: CONFIRMED from standard FJSSP
- Energy parameters: CONFIRMED from E-DFJSP 2025
- Machine modes: CONFIRMED from E-DFJSP 2025
- Auxiliary energy: PROPOSED (rarely modeled)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class MachineState(Enum):
    """Machine operational states"""
    IDLE = "idle"
    PROCESSING = "processing"
    SETUP = "setup"
    OFF = "off"
    BROKEN = "broken"


@dataclass
class MachineMode:
    """
    Machine operating mode (speed/feed rate setting)

    Evidence: Machine modes from E-DFJSP 2025 [CONFIRMED]
    - Different modes affect processing time and power consumption
    - Trade-off between speed and energy
    """
    mode_id: int
    mode_name: str = ""

    # Processing characteristics
    speed_factor: float = 1.0  # Multiplier for processing time (<1 = faster)
    power_multiplier: float = 1.0  # Multiplier for power consumption

    # Tool wear characteristics (PROPOSED)
    tool_wear_rate: float = 1.0

    def __post_init__(self):
        """Validate mode parameters"""
        if self.speed_factor <= 0:
            raise ValueError("Speed factor must be positive")
        if self.power_multiplier < 0:
            raise ValueError("Power multiplier must be non-negative")


@dataclass
class Machine:
    """
    A machine in the flexible job shop

    In SFJSSP, machines have:
    - Multiple operating modes
    - Energy consumption in different states
    - Potential for breakdowns
    """
    machine_id: int
    machine_name: str = ""

    # Operating modes (CONFIRMED from E-DFJSP 2025)
    modes: List[MachineMode] = field(default_factory=list)
    default_mode_id: int = 0

    # Energy parameters in kW (CONFIRMED from E-DFJSP 2025)
    power_processing: float = 10.0  # Power during processing
    power_idle: float = 2.0       # Power during idle
    power_setup: float = 5.0     # Power during setup
    startup_energy: float = 50.0   # Energy to start from off (kWh)

    # Setup characteristics (CONFIRMED from E-DFJSP 2025)
    setup_time: float = 0.0  # Time to switch between jobs

    # --- ADDED: Transport characteristics (PDF Section 5.2.2) ---
    power_transport: float = 2.0  # Power during transport (kW)
    total_transport_time: float = 0.0

    # Auxiliary energy allocation (PROPOSED)
    # Portion of facility auxiliary energy (lighting, HVAC) allocated to this machine
    auxiliary_power_share: float = 0.0  # kW

    # Current state
    current_state: MachineState = MachineState.IDLE
    current_job: Optional[int] = None
    current_operation: Optional[int] = None
    current_mode: Optional[int] = None

    # Temporal tracking
    available_time: float = 0.0  # When machine becomes available
    total_processing_time: float = 0.0
    total_idle_time: float = 0.0
    total_setup_time: float = 0.0

    # Count how many times this machine has been started from OFF state
    startup_count: int = 0

    # Breakdown tracking (PROPOSED for dynamic scenarios)
    is_broken: bool = False
    breakdown_time: Optional[float] = None
    repair_time: Optional[float] = None

    def get_power(self, state: MachineState, mode_id: Optional[int] = None) -> float:
        """
        Get power consumption for a given state

        Evidence: State-dependent power from E-DFJSP 2025 [CONFIRMED]
        """
        if state == MachineState.PROCESSING:
            power = self.power_processing
            if mode_id is not None and self.modes:
                mode = next((m for m in self.modes if m.mode_id == mode_id), None)
                if mode:
                    power *= mode.power_multiplier
            return power
        elif state == MachineState.IDLE:
            return self.power_idle
        elif state == MachineState.SETUP:
            return self.power_setup
        elif state == MachineState.OFF:
            return 0.0
        elif state == MachineState.BROKEN:
            return self.power_idle  # Still consumes standby power
        else:
            return 0.0

    def get_processing_energy(self, duration: float, mode_id: Optional[int] = None) -> float:
        """Calculate energy for processing duration"""
        return self.get_power(MachineState.PROCESSING, mode_id) * duration

    def get_idle_energy(self, duration: float) -> float:
        """Calculate energy for idle duration"""
        return self.power_idle * duration

    def get_setup_energy(self, duration: float) -> float:
        """Calculate energy for setup duration"""
        return self.power_setup * duration

    def calculate_energy_consumption(self) -> Dict[str, float]:
        """
        Calculate total energy consumption by category

        Returns dict with keys: processing, idle, setup, startup, auxiliary, total
        """
        energy = {
            'processing': self.power_processing * self.total_processing_time, # EM
            'idle': self.power_idle * self.total_idle_time,                   # EM
            'setup': self.power_setup * self.total_setup_time,                # EM
            'transport': self.power_transport * self.total_transport_time,    # ET (ADDED)
            'startup': self.startup_energy * self.startup_count,              # FIXED
            'auxiliary': self.auxiliary_power_share * (                       # EC
                self.total_processing_time + self.total_idle_time + 
                self.total_setup_time + self.total_transport_time
            )
        }
        energy['total'] = sum(energy.values()) # min ET + EM + EC
        return energy

    def is_available(self, current_time: float) -> bool:
        """Check if machine is available at current time"""
        return not self.is_broken and self.available_time <= current_time

    def schedule_breakdown(self, breakdown_time: float, repair_duration: float):
        """Schedule a machine breakdown"""
        self.is_broken = True
        self.breakdown_time = breakdown_time
        self.repair_time = breakdown_time + repair_duration
        self.current_state = MachineState.BROKEN

    def repair(self, current_time: float):
        """Repair the machine"""
        if self.is_broken and current_time >= self.repair_time:
            self.is_broken = False
            self.current_state = MachineState.IDLE
            self.available_time = max(self.available_time, current_time)

    def reset(self):
        """Reset machine to initial state"""
        self.current_state = MachineState.IDLE
        self.current_job = None
        self.current_operation = None
        self.current_mode = None
        self.available_time = 0.0
        self.total_processing_time = 0.0
        self.total_idle_time = 0.0
        self.total_setup_time = 0.0
        self.total_transport_time = 0.0  # ADDED
        self.startup_count = 0           # ADDED
        self.is_broken = False
        self.breakdown_time = None
        self.repair_time = None

    def __hash__(self):
        return hash(self.machine_id)

    def __eq__(self, other):
        if not isinstance(other, Machine):
            return False
        return self.machine_id == other.machine_id