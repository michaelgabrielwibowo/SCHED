import pytest
import numpy as np
from sfjssp_model.job import Job, Operation
from sfjssp_model.machine import Machine, MachineMode, MachineState
from sfjssp_model.worker import Worker, WorkerState
from sfjssp_model.instance import SFJSSPInstance, InstanceType
from sfjssp_model.schedule import Schedule, ScheduledOperation
from baseline_solver.greedy_solvers import GreedyScheduler
from moea.nsga3 import evaluate_sfjssp_genome

def test_ergonomic_limit_value():
    """Verify OCRA max is strictly 2.2."""
    worker = Worker(worker_id=1)
    assert worker.ocra_max_per_shift == 2.2

def test_transport_energy_calculation():
    """Verify transport energy is calculated dynamically."""
    instance = SFJSSPInstance(instance_id="test_energy")
    # Machine with 3.0 kW transport power
    machine = Machine(machine_id=0, power_transport=3.0)
    instance.add_machine(machine)
    instance.add_worker(Worker(0))
    
    schedule = Schedule(instance)
    # Op with 4.0 transport time
    op = ScheduledOperation(job_id=0, op_id=0, machine_id=0, worker_id=0, mode_id=0,
                            start_time=0.0, completion_time=10.0, processing_time=10.0,
                            transport_time=4.0)
    schedule.scheduled_ops[(0, 0)] = op
    
    energy = schedule.compute_total_energy(instance)
    # Transport energy should be 3.0 * 4.0 = 12.0
    assert energy['transport'] == 12.0
    assert energy['total'] >= 12.0

def test_feasibility_transport_time():
    """Verify precedence feasibility check includes transport time."""
    instance = SFJSSPInstance(instance_id="test_feas")
    instance.add_machine(Machine(machine_id=0))
    instance.add_machine(Machine(machine_id=1))
    instance.add_worker(Worker(0))
    job = Job(job_id=0, operations=[
        Operation(job_id=0, op_id=0, processing_times={0: {0: 10.0}}, eligible_machines={0}, eligible_workers={0}),
        Operation(job_id=0, op_id=1, processing_times={1: {0: 10.0}}, eligible_machines={1}, eligible_workers={0})
    ])
    instance.add_job(job)
    
    schedule = Schedule(instance)
    # Op 0 ends at 10.0, has 5.0 transport time.
    schedule.add_operation(job_id=0, op_id=0, machine_id=0, worker_id=0, mode_id=0,
                           start_time=0.0, completion_time=10.0, processing_time=10.0,
                           transport_time=5.0)
    
    # Op 1 starts at 14.0 (Violation: 10.0 + 5.0 = 15.0 required)
    schedule.add_operation(job_id=0, op_id=1, machine_id=1, worker_id=0, mode_id=0,
                           start_time=14.0, completion_time=24.0, processing_time=10.0)
    
    assert schedule.check_feasibility(instance) is False
    assert any("Precedence violation" in v for v in schedule.constraint_violations)

def test_feasibility_setup_time():
    """Verify machine overlap feasibility check includes setup time."""
    instance = SFJSSPInstance(instance_id="test_feas_setup")
    instance.add_machine(Machine(machine_id=0))
    instance.add_worker(Worker(0))
    job0 = Job(job_id=0, operations=[Operation(job_id=0, op_id=0, processing_times={0: {0: 10.0}}, eligible_machines={0}, eligible_workers={0})])
    job1 = Job(job_id=1, operations=[Operation(job_id=1, op_id=0, processing_times={0: {0: 10.0}}, eligible_machines={0}, eligible_workers={0})])
    instance.add_job(job0)
    instance.add_job(job1)
    
    schedule = Schedule(instance)
    # Op (0,0) ends at 10.0
    schedule.add_operation(job_id=0, op_id=0, machine_id=0, worker_id=0, mode_id=0,
                           start_time=0.0, completion_time=10.0, processing_time=10.0)
    
    # Op (1,0) starts at 14.0, has 5.0 setup time (Violation: 10.0 + 5.0 = 15.0 required)
    schedule.add_operation(job_id=1, op_id=0, machine_id=0, worker_id=0, mode_id=0,
                           start_time=14.0, completion_time=24.0, processing_time=10.0,
                           setup_time=5.0)
    
    assert schedule.check_feasibility(instance) is False
    assert any("Machine overlap" in v for v in schedule.constraint_violations)

def test_greedy_mandatory_rest():
    """Test that GreedyScheduler enforces mandatory rest."""
    # Create instance where a worker works a lot and needs rest
    instance = SFJSSPInstance(instance_id="test_rest")
    instance.add_machine(Machine(machine_id=0, power_processing=10.0))
    # Worker with 50% rest rule for easier testing
    worker = Worker(0, base_efficiency=1.0, labor_cost_per_hour=20.0, min_rest_fraction=0.5)
    instance.add_worker(worker)
    
    for i in range(5):
        instance.add_job(Job(i, operations=[Operation(job_id=i, op_id=0, processing_times={0: {0: 10.0}}, eligible_machines={0}, eligible_workers={0})]))
        # Set low risk rate (0.001) to avoid OCRA lockout
        instance.ergonomic_risk_map[(i, 0)] = 0.001
    
    scheduler = GreedyScheduler()
    schedule = scheduler.schedule(instance)
    
    # Check if any operation was delayed for rest
    # Total work = 5 * 10 = 50. With 50% rest rule, total time must be at least 100.
    assert schedule.makespan >= 100.0

def test_nsga3_penalty():
    """Verify NSGA-III evaluation applies penalty for high OCRA."""
    instance = SFJSSPInstance(instance_id="test_penalty")
    instance.add_machine(Machine(machine_id=0, power_processing=10.0))
    instance.add_worker(Worker(0))
    instance.add_job(Job(0, operations=[Operation(job_id=0, op_id=0, processing_times={0: {0: 100.0}}, eligible_machines={0}, eligible_workers={0})]))
    
    # Create genome that will cause high OCRA
    # OCRA risk rate defaults to 0.5 in code if not mapped. 0.5 * 100 = 50.0.
    # 50.0 > 2.2, so penalty should trigger.
    genome = {
        'sequence': np.array([0]),
        'machines': np.array([0]),
        'workers': np.array([0]),
        'offsets': np.array([0]),
        'op_list': [(0, 0)]
    }
    
    objectives = evaluate_sfjssp_genome(instance, genome)
    # objectives = [makespan, energy, ocra, labor]
    # makespan = 100.0 + penalty (1e6 * (50.0 - 2.2))
    assert objectives[0] > 1000000.0
