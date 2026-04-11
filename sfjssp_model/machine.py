"""
Machine and Machine Mode data structures for SFJSSP

Evidence Status:
- Basic machine structure: CONFIRMED from standard FJSSP
- Energy parameters: CONFIRMED from E-DFJSP 2025
- Machine modes: CONFIRMED from E-DFJSP 2025
- Auxiliary energy: PROPOSED (rarely modeled)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
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
    speed_factor: float = 1.0  # Processing-speed multiplier (>1 = faster)
    power_multiplier: float = 1.0  # Multiplier for power consumption

    # Tool wear characteristics (PROPOSED)
    tool_wear_rate: float = 1.0

    def to_dict(self) -> dict:
        """Convert mode to dictionary for serialization"""
        return {
            'mode_id': self.mode_id,
            'mode_name': self.mode_name,
            'speed_factor': self.speed_factor,
            'power_multiplier': self.power_multiplier,
            'tool_wear_rate': self.tool_wear_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MachineMode':
        """Create mode from dictionary"""
        return cls(
            mode_id=data['mode_id'],
            mode_name=data.get('mode_name', ""),
            speed_factor=data.get('speed_factor', 1.0),
            power_multiplier=data.get('power_multiplier', 1.0),
            tool_wear_rate=data.get('tool_wear_rate', 1.0),
        )

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

    # Tool wear state (INDUSTRY 5.0 Resilience)
    tool_health: float = 1.0  # 1.0 = new, 0.0 = broken
    
    def degrade_tool(self, duration: float, mode_id: Optional[int] = None):
        """
        Apply tool wear based on duration and mode.
        """
        wear_rate = 0.0001 # Base wear
        if mode_id is not None and self.modes:
            mode = next((m for m in self.modes if m.mode_id == mode_id), None)
            if mode:
                wear_rate *= mode.tool_wear_rate
        
        self.tool_health = max(0.0, self.tool_health - (wear_rate * duration))

    def to_dict(self) -> dict:
        """Convert machine to dictionary for serialization"""
        return {
            'machine_id': self.machine_id,
            'machine_name': self.machine_name,
            'modes': [m.to_dict() for m in self.modes],
            'default_mode_id': self.default_mode_id,
            'power_processing': self.power_processing,
            'power_idle': self.power_idle,
            'power_setup': self.power_setup,
            'startup_energy': self.startup_energy,
            'setup_time': self.setup_time,
            'power_transport': self.power_transport,
            'auxiliary_power_share': self.auxiliary_power_share,
            'current_state': self.current_state.value,
            'current_job': self.current_job,
            'current_operation': self.current_operation,
            'current_mode': self.current_mode,
            'available_time': self.available_time,
            'total_processing_time': self.total_processing_time,
            'total_idle_time': self.total_idle_time,
            'total_setup_time': self.total_setup_time,
            'total_transport_time': self.total_transport_time,
            'startup_count': self.startup_count,
            'is_broken': self.is_broken,
            'breakdown_time': self.breakdown_time,
            'repair_time': self.repair_time,
            'tool_health': self.tool_health,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Machine':
        """Create machine from dictionary"""
        m = cls(
            machine_id=data['machine_id'],
            machine_name=data.get('machine_name', ""),
            default_mode_id=data.get('default_mode_id', 0),
            power_processing=data.get('power_processing', 10.0),
            power_idle=data.get('power_idle', 2.0),
            power_setup=data.get('power_setup', 5.0),
            startup_energy=data.get('startup_energy', 50.0),
            setup_time=data.get('setup_time', 0.0),
            power_transport=data.get('power_transport', 2.0),
            auxiliary_power_share=data.get('auxiliary_power_share', 0.0),
        )
        m.modes = [MachineMode.from_dict(mode_data) for mode_data in data.get('modes', [])]
        m.current_state = MachineState(data.get('current_state', 'idle'))
        m.current_job = data.get('current_job')
        m.current_operation = data.get('current_operation')
        m.current_mode = data.get('current_mode')
        m.available_time = data.get('available_time', 0.0)
        m.total_processing_time = data.get('total_processing_time', 0.0)
        m.total_idle_time = data.get('total_idle_time', 0.0)
        m.total_setup_time = data.get('total_setup_time', 0.0)
        m.total_transport_time = data.get('total_transport_time', 0.0)
        m.startup_count = data.get('startup_count', 0)
        m.is_broken = data.get('is_broken', False)
        m.breakdown_time = data.get('breakdown_time')
        m.repair_time = data.get('repair_time')
        m.tool_health = data.get('tool_health', 1.0)
        return m

    @staticmethod
    def _minutes_to_hours(duration: float) -> float:
        """Convert durations stored in minutes to hours for energy math."""
        return duration / 60.0

    def get_mode(self, mode_id: int) -> Optional[MachineMode]:
        """Return a machine mode by identifier."""
        return next((mode for mode in self.modes if mode.mode_id == mode_id), None)

    def get_processing_duration(self, base_time: float, mode_id: Optional[int] = None) -> float:
        """Adjust a base processing time by the selected machine mode."""
        mode = self.get_mode(mode_id) if mode_id is not None else None
        if mode is None:
            return base_time
        return base_time / mode.speed_factor

    def validate_gap(self, start_time: float, setup_duration: float) -> Tuple[bool, float]:
        """
        Validate if the machine has enough gap for an incoming operation.
        Must accommodate current available_time plus setup duration.
        """
        if self.is_broken:
            return False, float('inf')
        earliest_start = self.available_time + setup_duration
        if earliest_start <= start_time:
            return True, start_time
        return False, earliest_start

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
        return self.get_power(MachineState.PROCESSING, mode_id) * self._minutes_to_hours(duration)

    def get_idle_energy(self, duration: float) -> float:
        """Calculate energy for idle duration"""
        return self.power_idle * self._minutes_to_hours(duration)

    def get_setup_energy(self, duration: float) -> float:
        """Calculate energy for setup duration"""
        return self.power_setup * self._minutes_to_hours(duration)

    def calculate_energy_consumption(self) -> Dict[str, float]:
        """
        Calculate total energy consumption by category

        Returns dict with keys: processing, idle, setup, startup, auxiliary, total
        """
        energy = {
            'processing': self.get_processing_energy(self.total_processing_time),  # kWh
            'idle': self.get_idle_energy(self.total_idle_time),                   # kWh
            'setup': self.get_setup_energy(self.total_setup_time),                # kWh
            'transport': self.power_transport * self._minutes_to_hours(self.total_transport_time),
            'startup': self.startup_energy * self.startup_count,              # FIXED
            'auxiliary': self.auxiliary_power_share * self._minutes_to_hours(
                self.total_processing_time
                + self.total_idle_time
                + self.total_setup_time
                + self.total_transport_time
            ),
        }
        energy['total'] = sum(energy.values()) # min ET + EM + EC
        return energy

    def is_available(self, current_time: float, ignore_temporal: bool = False) -> bool:
        """Check if machine is available at current time"""
        return not self.is_broken and (ignore_temporal or self.available_time <= current_time)

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
        self.tool_health = 1.0

    def __hash__(self):
        return hash(self.machine_id)

    def __eq__(self, other):
        if not isinstance(other, Machine):
            return False
        return self.machine_id == other.machine_id
