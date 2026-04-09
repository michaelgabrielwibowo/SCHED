"""
Tests for SFJSSP Core Model

Evidence Status:
- Test structure: Standard pytest conventions [CONFIRMED]
- Test coverage: Core data structures and constraints [PROPOSED]
"""

import pytest
import numpy as np

from sfjssp_model.job import Job, Operation
from sfjssp_model.machine import Machine, MachineMode, MachineState
from sfjssp_model.worker import Worker, WorkerState
from sfjssp_model.instance import SFJSSPInstance, InstanceType, DynamicEventParams
from sfjssp_model.schedule import Schedule, ScheduledOperation


class TestOperation:
    """Tests for Operation dataclass"""

    def test_operation_creation(self):
        """Test basic operation creation"""
        op = Operation(
            job_id=0,
            op_id=0,
            eligible_machines={0, 1},
            eligible_workers={0, 1},
        )

        assert op.job_id == 0
        assert op.op_id == 0
        assert not op.is_scheduled
        assert not op.is_completed

    def test_get_processing_time(self):
        """Test processing time retrieval"""
        op = Operation(
            job_id=0,
            op_id=0,
            processing_times={
                0: {0: 50.0, 1: 40.0},  # Machine 0: mode 0=50, mode 1=40
                1: {0: 60.0},  # Machine 1: mode 0=60
            },
            eligible_machines={0, 1},
            eligible_workers={0, 1},
        )

        # Test basic retrieval
        assert op.get_processing_time(0, 0) == 50.0
        assert op.get_processing_time(0, 1) == 40.0
        assert op.get_processing_time(1, 0) == 60.0

        # Test with worker efficiency
        assert op.get_processing_time(0, 0, worker_efficiency=0.8) == 62.5
        assert op.get_processing_time(0, 0, worker_efficiency=1.2) == pytest.approx(41.66666666666667)

    def test_get_processing_time_invalid(self):
        """Test processing time with invalid machine/mode"""
        op = Operation(
            job_id=0,
            op_id=0,
            processing_times={0: {0: 50.0}},
            eligible_machines={0},
            eligible_workers={0},
        )

        with pytest.raises(ValueError):
            op.get_processing_time(1, 0)  # Machine 1 not eligible

    def test_is_ready(self):
        """Test operation readiness check"""
        op0 = Operation(job_id=0, op_id=0, eligible_machines={0}, eligible_workers={0})
        op1 = Operation(job_id=0, op_id=1, eligible_machines={0}, eligible_workers={0})

        # First operation is always ready
        assert op0.is_ready(set())

        # Second operation needs predecessor
        assert not op1.is_ready(set())
        assert op1.is_ready({(0, 0)})


class TestJob:
    """Tests for Job dataclass"""

    def test_job_creation(self):
        """Test basic job creation"""
        job = Job(job_id=0)
        assert job.job_id == 0
        assert job.weight == 1.0
        assert job.arrival_time == 0.0

    def test_get_tardiness(self):
        """Test tardiness calculation"""
        job = Job(job_id=0, due_date=100.0)
        job.completion_time = 120.0

        assert job.get_tardiness() == 20.0

        # Early completion
        job.completion_time = 80.0
        assert job.get_tardiness() == 0.0

        # No due date
        job.due_date = None
        assert job.get_tardiness() == 0.0


class TestMachine:
    """Tests for Machine dataclass"""

    def test_machine_creation(self):
        """Test basic machine creation"""
        machine = Machine(machine_id=0)
        assert machine.machine_id == 0
        assert machine.current_state == MachineState.IDLE
        assert machine.available_time == 0.0

    def test_get_power(self):
        """Test power consumption by state"""
        machine = Machine(
            machine_id=0,
            power_processing=50.0,
            power_idle=5.0,
            power_setup=20.0,
        )

        assert machine.get_power(MachineState.PROCESSING) == 50.0
        assert machine.get_power(MachineState.IDLE) == 5.0
        assert machine.get_power(MachineState.SETUP) == 20.0
        assert machine.get_power(MachineState.OFF) == 0.0

    def test_get_power_with_mode(self):
        """Test power with mode multiplier"""
        mode = MachineMode(mode_id=0, speed_factor=1.0, power_multiplier=1.5)
        machine = Machine(
            machine_id=0,
            power_processing=50.0,
            modes=[mode],
        )

        assert machine.get_power(MachineState.PROCESSING, mode_id=0) == 75.0

    def test_energy_calculations(self):
        """Test energy calculation methods"""
        machine = Machine(
            machine_id=0,
            power_processing=50.0,
            power_idle=5.0,
        )

        assert machine.get_processing_energy(10.0) == 500.0
        assert machine.get_idle_energy(20.0) == 100.0

    def test_is_available(self):
        """Test machine availability check"""
        machine = Machine(machine_id=0, available_time=100.0)

        assert not machine.is_available(50.0)
        assert machine.is_available(100.0)
        assert machine.is_available(150.0)

    def test_schedule_breakdown(self):
        """Test machine breakdown scheduling"""
        machine = Machine(machine_id=0)

        machine.schedule_breakdown(100.0, 50.0)

        assert machine.is_broken
        assert machine.breakdown_time == 100.0
        assert machine.repair_time == 150.0
        assert machine.current_state == MachineState.BROKEN

    def test_repair(self):
        """Test machine repair"""
        machine = Machine(machine_id=0, available_time=100.0)
        machine.schedule_breakdown(100.0, 50.0)

        machine.repair(160.0)

        assert not machine.is_broken
        assert machine.current_state == MachineState.IDLE


class TestWorker:
    """Tests for Worker dataclass"""

    def test_worker_creation(self):
        """Test basic worker creation"""
        worker = Worker(worker_id=0)
        assert worker.worker_id == 0
        assert worker.base_efficiency == 1.0
        assert worker.fatigue_current == 0.0

    def test_get_efficiency(self):
        """Test efficiency calculation"""
        worker = Worker(
            worker_id=0,
            base_efficiency=1.0,
            fatigue_current=0.0,
        )

        assert worker.get_efficiency() == 1.0

        # Fatigue reduces efficiency
        worker.fatigue_current = 0.5
        assert worker.get_efficiency() < 1.0

    def test_update_fatigue(self):
        """Test fatigue dynamics"""
        worker = Worker(
            worker_id=0,
            fatigue_rate=0.03,  # alpha
            recovery_rate=0.05,  # beta
            fatigue_current=0.0,
        )

        # Work increases fatigue
        worker.update_fatigue(work_duration=10.0, rest_duration=0.0)
        assert worker.fatigue_current == pytest.approx(0.3)

        # Rest decreases fatigue
        worker.update_fatigue(work_duration=0.0, rest_duration=10.0)
        assert worker.fatigue_current < 0.3

    def test_add_ergonomic_risk(self):
        """Test ergonomic risk accumulation"""
        worker = Worker(worker_id=0)

        worker.add_ergonomic_risk(risk_rate=0.5, duration=10.0)
        assert worker.ocra_current_shift == 5.0

    def test_is_ergonomic_limit_exceeded(self):
        """Test ergonomic limit check"""
        worker = Worker(
            worker_id=0,
            ocra_max_per_shift=3.0,
        )

        assert not worker.is_ergonomic_limit_exceeded()

        worker.add_ergonomic_risk(risk_rate=0.5, duration=10.0)
        assert worker.is_ergonomic_limit_exceeded()

    def test_needs_rest(self):
        """Test rest requirement check"""
        worker = Worker(
            worker_id=0,
            max_consecutive_work_time=480.0,
            min_rest_fraction=0.125,
        )

        # No rest needed initially
        assert not worker.needs_rest()

        # After long work period
        worker.current_work_duration = 500.0
        assert worker.needs_rest()

    def test_get_labor_cost(self):
        """Test labor cost calculation"""
        worker = Worker(worker_id=0, labor_cost_per_hour=20.0)

        assert worker.get_labor_cost(8.0) == 160.0


class TestSFJSSPInstance:
    """Tests for SFJSSPInstance dataclass"""

    def test_instance_creation(self):
        """Test basic instance creation"""
        instance = SFJSSPInstance(instance_id="TEST_001")
        assert instance.instance_id == "TEST_001"
        assert instance.n_jobs == 0
        assert instance.n_machines == 0
        assert instance.n_workers == 0

    def test_add_entities(self):
        """Test adding jobs, machines, workers"""
        instance = SFJSSPInstance(instance_id="TEST_001")

        # Add machine
        machine = Machine(machine_id=0)
        instance.add_machine(machine)
        assert instance.n_machines == 1

        # Add worker
        worker = Worker(worker_id=0)
        instance.add_worker(worker)
        assert instance.n_workers == 1

        # Add job
        job = Job(job_id=0)
        instance.add_job(job)
        assert instance.n_jobs == 1

    def test_get_entities(self):
        """Test entity retrieval"""
        instance = SFJSSPInstance(instance_id="TEST_001")

        machine = Machine(machine_id=5)
        instance.add_machine(machine)

        assert instance.get_machine(5).machine_id == 5
        assert instance.get_machine(99) is None

    def test_get_eligible_resources(self):
        """Test eligibility query"""
        instance = SFJSSPInstance(instance_id="TEST_001")

        # Setup
        machine0 = Machine(machine_id=0)
        machine1 = Machine(machine_id=1)
        instance.add_machine(machine0)
        instance.add_machine(machine1)

        worker0 = Worker(worker_id=0, eligible_operations={(0, 0)})
        instance.add_worker(worker0)

        op = Operation(
            job_id=0, op_id=0,
            eligible_machines={0},
            eligible_workers={0},
        )
        job = Job(job_id=0, operations=[op])
        instance.add_job(job)

        assert instance.get_eligible_machines(0, 0) == [0]
        assert instance.get_eligible_workers(0, 0) == [0]

    def test_dynamic_job_generation(self):
        """Test dynamic job arrival generation"""
        instance = SFJSSPInstance(
            instance_id="TEST_DYNAMIC",
            instance_type=InstanceType.DYNAMIC,
            dynamic_params=DynamicEventParams(arrival_rate=1.0),
        )

        # Add resources for job generation
        for i in range(5):
            instance.add_machine(Machine(machine_id=i))
            instance.add_worker(Worker(worker_id=i))

        rng = np.random.default_rng(42)

        # Generate some jobs
        jobs_generated = 0
        for _ in range(100):
            job = instance.generate_dynamic_job(0.0, rng)
            if job:
                jobs_generated += 1

        # With high arrival rate, should generate some jobs
        assert jobs_generated > 0

    def test_to_dict(self):
        """Test instance serialization"""
        instance = SFJSSPInstance(
            instance_id="TEST_001"
        )
        instance.add_job(Job(0))

        data = instance.to_dict()

        assert data['instance_id'] == "TEST_001"
        assert len(data['jobs']) == 1
        assert 'label' in data


class TestSchedule:
    """Tests for Schedule dataclass"""

    def test_schedule_creation(self):
        """Test basic schedule creation"""
        schedule = Schedule(instance_id="TEST_001")
        assert schedule.instance_id == "TEST_001"
        assert len(schedule.scheduled_ops) == 0
        assert schedule.is_feasible

    def test_add_operation(self):
        """Test adding scheduled operation"""
        schedule = Schedule(instance_id="TEST_001")

        schedule.add_operation(
            job_id=0,
            op_id=0,
            machine_id=0,
            worker_id=0,
            mode_id=0,
            start_time=0.0,
            completion_time=50.0,
            processing_time=50.0,
        )

        assert schedule.is_operation_scheduled(0, 0)
        assert not schedule.is_operation_scheduled(0, 1)

    def test_get_operation(self):
        """Test operation retrieval"""
        schedule = Schedule(instance_id="TEST_001")

        schedule.add_operation(
            job_id=0, op_id=0,
            machine_id=0, worker_id=0, mode_id=0,
            start_time=0.0, completion_time=50.0, processing_time=50.0,
        )

        op = schedule.get_operation(0, 0)
        assert op is not None
        assert op.machine_id == 0
        assert op.worker_id == 0

    def test_compute_makespan(self):
        """Test makespan calculation"""
        schedule = Schedule(instance_id="TEST_001")

        schedule.add_operation(
            job_id=0, op_id=0,
            machine_id=0, worker_id=0, mode_id=0,
            start_time=0.0, completion_time=50.0, processing_time=50.0,
        )
        schedule.add_operation(
            job_id=1, op_id=0,
            machine_id=1, worker_id=1, mode_id=0,
            start_time=0.0, completion_time=100.0, processing_time=100.0,
        )

        makespan = schedule.compute_makespan()
        assert makespan == 100.0

    def test_get_job_tardiness(self):
        """Test tardiness calculation in schedule context"""
        instance = SFJSSPInstance(instance_id="TEST_001")
        op = Operation(job_id=0, op_id=0, eligible_machines={0}, eligible_workers={0})
        job = Job(job_id=0, due_date=100.0, operations=[op])
        instance.add_job(job)

        schedule = Schedule(instance_id="TEST_001")
        schedule.add_operation(
            job_id=0, op_id=0,
            machine_id=0, worker_id=0, mode_id=0,
            start_time=0.0, completion_time=120.0, processing_time=120.0,
        )

        tardiness = schedule.get_job_tardiness(0, instance)
        assert tardiness == 20.0

    def test_check_feasibility_precedence(self):
        """Test feasibility check - precedence constraints"""
        instance = SFJSSPInstance(instance_id="TEST_001")
        op0 = Operation(job_id=0, op_id=0, eligible_machines={0}, eligible_workers={0})
        op1 = Operation(job_id=0, op_id=1, eligible_machines={0}, eligible_workers={0})
        job = Job(job_id=0, operations=[op0, op1])
        instance.add_job(job)
        instance.add_machine(Machine(machine_id=0))
        instance.add_worker(Worker(worker_id=0))

        schedule = Schedule(instance_id="TEST_001")

        # Violate precedence: op1 before op0
        schedule.add_operation(
            job_id=0, op_id=1,
            machine_id=0, worker_id=0, mode_id=0,
            start_time=0.0, completion_time=50.0, processing_time=50.0,
        )
        schedule.add_operation(
            job_id=0, op_id=0,
            machine_id=0, worker_id=0, mode_id=0,
            start_time=60.0, completion_time=110.0, processing_time=50.0,
        )

        feasible = schedule.check_feasibility(instance)
        assert not feasible
        assert any("Precedence" in v for v in schedule.constraint_violations)

    def test_check_feasibility_machine_overlap(self):
        """Test feasibility check - machine capacity"""
        instance = SFJSSPInstance(instance_id="TEST_001")

        # Two operations on same machine with overlap
        op0 = Operation(job_id=0, op_id=0, eligible_machines={0}, eligible_workers={0})
        op1 = Operation(job_id=1, op_id=0, eligible_machines={0}, eligible_workers={1})
        instance.add_job(Job(job_id=0, operations=[op0]))
        instance.add_job(Job(job_id=1, operations=[op1]))
        instance.add_machine(Machine(machine_id=0))
        instance.add_worker(Worker(worker_id=0))
        instance.add_worker(Worker(worker_id=1))

        schedule = Schedule(instance_id="TEST_001")

        # Overlapping operations on same machine
        schedule.add_operation(
            job_id=0, op_id=0,
            machine_id=0, worker_id=0, mode_id=0,
            start_time=0.0, completion_time=100.0, processing_time=100.0,
        )
        schedule.add_operation(
            job_id=1, op_id=0,
            machine_id=0, worker_id=1, mode_id=0,
            start_time=50.0, completion_time=150.0, processing_time=100.0,
        )

        feasible = schedule.check_feasibility(instance)
        assert not feasible
        assert any("Machine overlap" in v for v in schedule.constraint_violations)

    def test_evaluate(self):
        """Test schedule evaluation"""
        instance = SFJSSPInstance(instance_id="TEST_001")

        # Setup
        machine = Machine(
            machine_id=0,
            power_processing=50.0,
            power_idle=5.0,
        )
        instance.add_machine(machine)
        instance.add_worker(Worker(worker_id=0))

        job = Job(job_id=0, due_date=100.0)
        op = Operation(job_id=0, op_id=0, eligible_machines={0}, eligible_workers={0})
        op.processing_times[0] = {0: 50.0}
        job.operations = [op]
        instance.add_job(job)
        instance.ergonomic_risk_map[(0, 0)] = 0.3

        schedule = Schedule(instance_id="TEST_001")
        schedule.add_operation(
            job_id=0, op_id=0,
            machine_id=0, worker_id=0, mode_id=0,
            start_time=0.0, completion_time=50.0, processing_time=50.0,
        )

        objectives = schedule.evaluate(instance)

        assert 'makespan' in objectives
        assert 'total_energy' in objectives
        assert objectives['makespan'] == 50.0


class TestIntegration:
    """Integration tests for complete workflow"""

    def test_create_and_schedule_instance(self):
        """Test creating instance and generating schedule"""
        # Create instance
        instance = SFJSSPInstance(instance_id="INTEGRATION_TEST")

        # Add machines
        for i in range(3):
            machine = Machine(
                machine_id=i,
                power_processing=30.0 + i * 10,
                power_idle=3.0 + i,
            )
            instance.add_machine(machine)

        # Add workers
        for i in range(2):
            worker = Worker(
                worker_id=i,
                labor_cost_per_hour=15.0 + i * 5,
                fatigue_rate=0.02 + i * 0.01,
            )
            instance.add_worker(worker)

        # Add jobs
        for job_id in range(5):
            ops = []
            for op_idx in range(2):
                op = Operation(
                    job_id=job_id,
                    op_id=op_idx,
                    eligible_machines={0, 1, 2},
                    eligible_workers={0, 1},
                )
                op.processing_times = {
                    0: {0: 30.0 + op_idx * 10},
                    1: {0: 40.0 + op_idx * 10},
                    2: {0: 35.0 + op_idx * 10},
                }
                ops.append(op)

            job = Job(
                job_id=job_id,
                operations=ops,
                due_date=200.0 + job_id * 50,
            )
            instance.add_job(job)

        # Create simple schedule (greedy assignment)
        schedule = Schedule(instance_id=instance.instance_id)
        current_time = 0.0

        for job in instance.jobs:
            for op in job.operations:
                # Pick first eligible machine and worker
                m_id = list(op.eligible_machines)[0]
                w_id = list(op.eligible_workers)[0]

                pt = op.get_processing_time(m_id, 0)
                schedule.add_operation(
                    job_id=job.job_id,
                    op_id=op.op_id,
                    machine_id=m_id,
                    worker_id=w_id,
                    mode_id=0,
                    start_time=current_time,
                    completion_time=current_time + pt,
                    processing_time=pt,
                )
                current_time += pt

        # Evaluate
        schedule.compute_makespan()
        schedule.check_feasibility(instance)
        objectives = schedule.evaluate(instance)

        assert schedule.makespan > 0
        assert objectives['total_energy'] > 0
        assert objectives['total_labor_cost'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
