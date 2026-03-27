import pytest
import numpy as np

from sfjssp_model.job import Job, Operation
from sfjssp_model.machine import Machine, MachineMode, MachineState
from sfjssp_model.worker import Worker, WorkerState
from sfjssp_model.instance import SFJSSPInstance, InstanceType
from sfjssp_model.schedule import Schedule, ScheduledOperation
from baseline_solver.greedy_solvers import GreedyScheduler
from exact_solvers.cp_solver import EnergyAwareCPScheduler

def test_dynamic_fatigue_recovery():
    """Test that a worker recovers fatigue correctly when resting."""
    worker = Worker(worker_id=0, base_efficiency=1.0, labor_cost_per_hour=20.0,
                    fatigue_rate=0.05, recovery_rate=0.1)
    
    # Initial state
    assert worker.fatigue_current == 0.0
    
    # Update fatigue directly to simulate work
    worker.update_fatigue(10.0, 0.0)  # Works for 10 units of time, rests 0
    
    expected_fatigue = 10.0 * 0.05
    assert pytest.approx(worker.fatigue_current) == expected_fatigue
    
    # Now simulate a rest period
    rest_duration = 2.0
    worker.record_rest(rest_duration)
    
    # Expected fatigue after rest
    expected_fatigue_after_rest = max(0.0, expected_fatigue - (rest_duration * 0.1))
    assert pytest.approx(worker.fatigue_current) == expected_fatigue_after_rest

def test_compute_resilience_metrics():
    """Test that resilience metrics are properly calculated."""
    # Create simple instance
    instance = SFJSSPInstance(instance_id="test_res", instance_type=InstanceType.STATIC,
                              n_jobs=2, n_machines=2, n_workers=2)
                              
    # Add machines
    instance.machines.append(Machine(machine_id=0, machine_name="M0", modes=[MachineMode(0, 10, 5, 2)], current_state=MachineState.IDLE))
    instance.machines.append(Machine(machine_id=1, machine_name="M1", modes=[MachineMode(0, 10, 5, 2)], current_state=MachineState.IDLE))
    
    # Add workers
    instance.workers.append(Worker(0))
    instance.workers.append(Worker(1))
    
    # Add jobs
    instance.jobs.append(Job(job_id=0, due_date=50.0, weight=1.0, operations=[Operation(0, 0, {0: {0: 10.0}}, {0, 1}, {0, 1})]))
    instance.jobs.append(Job(job_id=1, due_date=60.0, weight=1.0, operations=[Operation(1, 0, {1: {0: 15.0}}, {0, 1}, {0, 1})]))
    
    schedule = Schedule(instance)
    
    # Schedule ops directly
    schedule.add_operation(job_id=0, op_id=0, machine_id=0, worker_id=0,
                           start_time=0.0, completion_time=10.0, processing_time=10.0, mode_id=0)
                           
    schedule.add_operation(job_id=1, op_id=0, machine_id=1, worker_id=1, # DIFFERENT WORKER
                           start_time=10.0, completion_time=25.0, processing_time=15.0, mode_id=0)
    
    metrics = schedule.compute_resilience_metrics(instance)
    
    assert "machine_workload_variance" in metrics
    assert "worker_workload_variance" in metrics
    assert "average_slack_time" in metrics
    
    # Worker 0 worked 10, Worker 1 worked 15. Variance is calculated from workloads [10, 15].
    # Mean is 12.5. Variance = ((10-12.5)^2 + (15-12.5)^2) / 2 = 6.25
    assert pytest.approx(metrics["worker_workload_variance"]) == np.var([10.0, 15.0])
    
    # Machine 0 worked 10, Machine 1 worked 15.
    assert pytest.approx(metrics["machine_workload_variance"]) == np.var([10.0, 15.0])
    
    # Slack time: Only one operation per job so no slack between operations
    assert pytest.approx(metrics["average_slack_time"]) == 0.0
    
def test_energy_objective_cp_solver():
    """Test that EnergyAwareCPScheduler can be initialized with different objectives."""
    instance = SFJSSPInstance(instance_id="test_energy", instance_type=InstanceType.STATIC,
                              n_jobs=1, n_machines=1, n_workers=1)
    # Just testing initialization and structure, not full solve here as it might be heavy.
    solver = EnergyAwareCPScheduler(time_limit=1, peak_power_penalty=50.0)
    
    assert solver.peak_power_penalty == 50.0

def test_full_evaluation_includes_resilience():
    """Test if Schedule.evaluate() returns the new resilience metrics."""
    instance = SFJSSPInstance(instance_id="test_res", instance_type=InstanceType.STATIC,
                              n_jobs=1, n_machines=1, n_workers=1)
    instance.machines.append(Machine(machine_id=0, machine_name="M0", modes=[MachineMode(0, 10, 5, 2)], current_state=MachineState.IDLE))
    instance.workers.append(Worker(0))
    instance.jobs.append(Job(job_id=0, due_date=50.0, weight=1.0, operations=[Operation(0, 0, {0: {0: 10.0}}, {0}, {0})]))
    
    schedule = Schedule(instance)
    schedule.add_operation(job_id=0, op_id=0, machine_id=0, worker_id=0,
                           start_time=0.0, completion_time=10.0, processing_time=10.0, mode_id=0)
    
    eval_metrics = schedule.evaluate(instance)
    
    # Assert top-level inclusion
    assert "machine_workload_variance" in eval_metrics
    assert "worker_workload_variance" in eval_metrics
    assert "average_slack_time" in eval_metrics
