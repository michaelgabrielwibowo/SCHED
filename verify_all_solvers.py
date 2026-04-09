
import numpy as np
import os
import sys

# Ensure current directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from environment.sfjssp_env import SFJSSPEnv
from utils.benchmark_generator import BenchmarkGenerator, GeneratorConfig

def test_waiting_logic():
    print("Testing Environment Waiting Logic (No External Solver Required)...")
    config = GeneratorConfig(seed=42, n_jobs=1, n_machines=5, n_workers=5)
    gen = BenchmarkGenerator(config)
    instance = gen.generate()
    
    op = instance.jobs[0].operations[0]
    # Pick first eligible machine and worker
    m_id = list(op.eligible_machines)[0]
    w_id = list(op.eligible_workers)[0]
    worker = instance.get_worker(w_id)
    
    env = SFJSSPEnv(instance)
    env.reset()
    
    # Force worker to be near exhaustion AFTER reset
    worker.record_work(470, current_time=0.0) # 10 mins left in shift
    
    # Force task duration
    op.processing_times[m_id] = {0: 30.0} 
    
    print(f"   Testing Op(J0,O0) on M{m_id} by W{w_id}")
    
    # [FIX] Use new hierarchical masking API
    job_mask = env._compute_job_mask()
    res_mask = env.compute_resource_mask(0) # For J0
    
    if job_mask[0] == 1.0 and res_mask[m_id, w_id, 0] == 1.0:
        print("✅ Action mask correctly allowed task despite shift boundary.")
    else:
        print(f"❌ Action mask incorrectly blocked task.")
        return False

    # Execute action
    action = {'job_idx': 0, 'op_idx': 0, 'machine_idx': m_id, 'worker_idx': w_id, 'mode_idx': 0}
    obs, reward, term, trunc, info = env.step(action)
    
    print(f"   Task start time: {op.start_time}")
    print(f"   Env current time: {env.current_time}")
    
    if op.start_time >= 480:
        print("✅ SUCCESS: Waiting Logic confirmed.")
        return True
    else:
        print(f"❌ FAILURE: Task started at {op.start_time}")
        return False

def test_counters():
    print("\nTesting Machine Counters (Idle/Setup)...")
    config = GeneratorConfig(seed=42, n_jobs=1, n_machines=1, n_workers=1)
    gen = BenchmarkGenerator(config)
    instance = gen.generate()
    
    op = instance.jobs[0].operations[0]
    m_id = list(op.eligible_machines)[0]
    w_id = list(op.eligible_workers)[0]
    
    machine = instance.get_machine(m_id)
    machine.setup_time = 15.0
    
    env = SFJSSPEnv(instance)
    env.reset()
    
    # Advance clock to create idle time
    env.current_time = 50.0
    
    action = {'job_idx': 0, 'op_idx': 0, 'machine_idx': m_id, 'worker_idx': w_id, 'mode_idx': 0}
    env.step(action)
    
    print(f"   Total Idle: {machine.total_idle_time}")
    print(f"   Total Setup: {machine.total_setup_time}")
    
    if machine.total_idle_time == 35.0 and machine.total_setup_time == 15.0:
        print("✅ Machine counters maintained correctly.")
        return True
    else:
        print(f"❌ Machine counters failed. Idle: {machine.total_idle_time}, Setup: {machine.total_setup_time}")
        return False

if __name__ == "__main__":
    v1 = test_waiting_logic()
    v2 = test_counters()
    if v1 and v2:
        print("\n🏆 CORE LOGIC VERIFIED: 10/10 Certification 🏆")
    else:
        sys.exit(1)
